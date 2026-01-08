# import jax
# import jax.numpy as jnp
# import flax.linen as nn
# import numpy as np
# from flax.linen.initializers import constant, orthogonal
# from typing import Sequence
# import distrax


# def get_activation_fn(name: str):
#     """ Get activation function by name from the flax.linen module."""
#     try:
#         # Use getattr to dynamically retrieve the activation function from jax.nn
#         return getattr(nn, name)
#     except AttributeError:
#         raise ValueError(f"Activation function '{name}' not found. Name must be the same as in flax.linen!")


# class FullyConnectedNet(nn.Module):

#     hidden_layer_dims: Sequence[int]
#     output_dim: int
#     activation: str = "tanh"
#     output_activation: str = None    # none means linear activation
#     use_running_mean_stand: bool = True
#     squeeze_output: bool = True

#     def setup(self):
#         self.activation_fn = get_activation_fn(self.activation)
#         self.output_activation_fn = get_activation_fn(self.output_activation) \
#             if self.output_activation is not None else lambda x: x

#     @nn.compact
#     def __call__(self, x):

#         if self.use_running_mean_stand:
#             x = RunningMeanStd()(x)

#         # build network
#         for i, dim_layer in enumerate(self.hidden_layer_dims):
#             x = nn.Dense(dim_layer, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
#             x = self.activation_fn(x)

#         # add last layer
#         x = nn.Dense(self.output_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
#         x = self.output_activation_fn(x)

#         return jnp.squeeze(x) if self.squeeze_output else x


# class ActorCritic(nn.Module):
#     action_dim: Sequence[int]
#     activation: str = "tanh"
#     init_std: float = 1.0
#     learnable_std: bool = True
#     hidden_layer_dims: Sequence[int] = (1024, 512)
#     actor_obs_ind: jnp.ndarray = None
#     critic_obs_ind: jnp.ndarray = None

#     def setup(self):
#         self.activation_fn = get_activation_fn(self.activation)

#     @nn.compact
#     def __call__(self, x):

#         x = RunningMeanStd()(x)

#         # build actor
#         actor_x = x if self.actor_obs_ind is None else x[..., self.actor_obs_ind]
#         actor_mean = FullyConnectedNet(self.hidden_layer_dims, self.action_dim, self.activation,
#                                        None, False, False)(actor_x)
#         actor_logtstd = self.param("log_std", nn.initializers.constant(jnp.log(self.init_std)),
#                                    (self.action_dim,))
#         if not self.learnable_std:
#             actor_logtstd = jax.lax.stop_gradient(actor_logtstd)

#         pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

#         # build critic
#         critic_x = x if self.critic_obs_ind is None else x[..., self.critic_obs_ind]
#         critic = FullyConnectedNet(self.hidden_layer_dims, 1, self.activation, None, False, False)(critic_x)

#         return pi, jnp.squeeze(critic, axis=-1)


# class RunningMeanStd(nn.Module):
#     """Layer that maintains running mean and variance for input normalization."""

#     @nn.compact
#     def __call__(self, x):

#         x = jnp.atleast_2d(x)

#         # Initialize running mean, variance, and count
#         mean = self.variable('run_stats', 'mean', lambda: jnp.zeros(x.shape[-1]))
#         var = self.variable('run_stats', 'var', lambda: jnp.ones(x.shape[-1]))
#         count = self.variable('run_stats', 'count', lambda: jnp.array(1e-6))

#         # Compute batch mean and variance
#         batch_mean = jnp.mean(x, axis=0)
#         batch_var = jnp.var(x, axis=0) + 1e-6  # Add epsilon for numerical stability
#         batch_count = x.shape[0]

#         # Update counts
#         updated_count = count.value + batch_count

#         # Numerically stable mean and variance update
#         delta = batch_mean - mean.value
#         new_mean = mean.value + delta * batch_count / updated_count

#         # Compute the new variance using Welford's method
#         m_a = var.value * count.value
#         m_b = batch_var * batch_count
#         M2 = m_a + m_b + jnp.square(delta) * count.value * batch_count / updated_count
#         new_var = M2 / updated_count

#         # Normalize input
#         normalized_x = (x - new_mean) / jnp.sqrt(new_var + 1e-8)

#         # Update state variables
#         mean.value = new_mean
#         var.value = new_var
#         count.value = updated_count

#         return jnp.squeeze(normalized_x)


# 11-20 残差网络加注意力机制改进
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence
import distrax


def get_activation_fn(name: str):
    """ Get activation function by name from the flax.linen module."""
    try:
        # Use getattr to dynamically retrieve the activation function from jax.nn
        return getattr(nn, name)
    except AttributeError:
        raise ValueError(f"Activation function '{name}' not found. Name must be the same as in flax.linen!")


class ResidualBlock(nn.Module):
    """Residual block with skip connections."""
    hidden_dim: int
    activation: str = "relu"
    use_projection: bool = False
    
    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)
    
    @nn.compact
    def __call__(self, x):
        residual = x
        
        # First linear layer
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = self.activation_fn(x)
        
        # Second linear layer
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        
        # Projection for dimension matching if needed
        if self.use_projection and residual.shape[-1] != self.hidden_dim:
            residual = nn.Dense(self.hidden_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(residual)
        
        # Skip connection
        x = x + residual
        x = self.activation_fn(x)
        
        return x


class SelfAttention(nn.Module):
    """Simple self-attention mechanism."""
    embed_dim: int
    num_heads: int = 8
    
    @nn.compact
    def __call__(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        # Multi-head attention
        attention = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=embed_dim,
            out_features=embed_dim,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0)
        )
        
        # Add position encoding if sequence length > 1
        if seq_len > 1:
            pos_embedding = self.param('pos_embedding', 
                                     nn.initializers.normal(stddev=0.02),
                                     (seq_len, embed_dim))
            x = x + pos_embedding
        
        attended = attention(x, x)
        
        # Feed-forward network
        x = nn.Dense(embed_dim * 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(attended)
        x = nn.relu(x)
        x = nn.Dense(embed_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        
        # Residual connection
        x = x + attended
        
        return x

# 12-8注意力机制改进
class ChannelAttention(nn.Module):
    """轻量级通道注意力机制 - 适合RL任务"""
    reduction_ratio: int = 16
    
    @nn.compact
    def __call__(self, x):
        # x shape: (batch, features)
        channels = x.shape[-1]
        
        # 确保至少有一个隐藏单元
        hidden_dim = max(channels // self.reduction_ratio, 8)
        
        # 全局统计
        # 对于2D输入，直接使用
        if len(x.shape) == 2:
            avg_pool = x  # (batch, channels)
            max_pool = x
        else:
            # 对于更高维度的输入
            avg_pool = jnp.mean(x, axis=tuple(range(1, len(x.shape)-1)), keepdims=False)
            max_pool = jnp.max(x, axis=tuple(range(1, len(x.shape)-1)), keepdims=False)
        
        # 共享MLP
        dense1 = nn.Dense(hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        dense2 = nn.Dense(channels, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        
        # 计算注意力权重
        avg_out = dense2(nn.relu(dense1(avg_pool)))
        max_out = dense2(nn.relu(dense1(max_pool)))
        
        # 注意力权重
        attention = nn.sigmoid(avg_out + max_out)
        
        # 应用注意力
        return x * attention


class BodyPartAttention(nn.Module):
    """针对身体部位的分组注意力机制"""
    embed_dim: int = 64
    num_heads: int = 4
    body_part_groups: dict = None  # {"legs": [indices], "arms": [indices], ...}
    
    @nn.compact
    def __call__(self, x, body_part_indices=None):
        """
        Args:
            x: 输入特征 (batch, obs_dim)
            body_part_indices: 身体部位索引字典，如果为None则使用全局注意力
        """
        if body_part_indices is None or len(body_part_indices) == 0:
            # 如果没有分组信息，使用全局注意力
            batch_size = x.shape[0]
            x_reshaped = x.reshape(batch_size, 1, -1)  # (batch, 1, features)
            
            # 投影到嵌入维度
            if x_reshaped.shape[-1] != self.embed_dim:
                x_proj = nn.Dense(self.embed_dim, 
                                 kernel_init=orthogonal(np.sqrt(2)), 
                                 bias_init=constant(0.0))(x_reshaped)
            else:
                x_proj = x_reshaped
            
            # 应用自注意力
            attention_out = SelfAttention(self.embed_dim, self.num_heads)(x_proj)
            
            # 恢复维度
            return attention_out.reshape(batch_size, -1)
        
        # 使用分组注意力
        batch_size = x.shape[0]
        attended_parts = []
        
        # 对每个身体部位应用注意力
        for part_name, indices in body_part_indices.items():
            if len(indices) == 0:
                continue
                
            # 提取该部位的特征
            part_features = x[..., indices]  # (batch, part_dim)
            part_dim = len(indices)
            
            # 重塑为序列形式
            part_seq = part_features.reshape(batch_size, 1, part_dim)
            
            # 投影到统一维度
            part_proj = nn.Dense(self.embed_dim,
                               kernel_init=orthogonal(np.sqrt(2)),
                               bias_init=constant(0.0),
                               name=f'proj_{part_name}')(part_seq)
            
            attended_parts.append(part_proj)
        
        if len(attended_parts) == 0:
            return x
        
        # 将所有部位拼接为序列
        parts_concat = jnp.concatenate(attended_parts, axis=1)  # (batch, num_parts, embed_dim)
        
        # 部位间交叉注意力
        cross_attended = SelfAttention(self.embed_dim, self.num_heads)(parts_concat)
        
        # 展平输出
        output = cross_attended.reshape(batch_size, -1)
        
        # 如果需要，投影回原始维度
        output_dim = x.shape[-1]
        if output.shape[-1] != output_dim:
            output = nn.Dense(output_dim,
                            kernel_init=orthogonal(np.sqrt(2)),
                            bias_init=constant(0.0))(output)
        
        return output


class ImprovedFullyConnectedNet(nn.Module):
    """改进的全连接网络，支持分组注意力和通道注意力"""
    hidden_layer_dims: Sequence[int]
    output_dim: int
    activation: str = "tanh"
    output_activation: str = None
    use_running_mean_stand: bool = True
    squeeze_output: bool = True
    use_residual: bool = False
    use_attention: bool = False
    attention_embed_dim: int = 64
    use_channel_attention: bool = False
    channel_attention_layers: Sequence[int] = (0, 2)  # 在哪些层后添加通道注意力
    channel_reduction_ratio: int = 16
    use_grouped_attention: bool = False
    body_part_indices: dict = None
    
    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)
        self.output_activation_fn = get_activation_fn(self.output_activation) \
            if self.output_activation is not None else lambda x: x

    @nn.compact
    def __call__(self, x):
        if self.use_running_mean_stand:
            x = RunningMeanStd()(x)

        # 阶段1：分组注意力（身体部位协调）
        if self.use_grouped_attention:
            x = BodyPartAttention(
                embed_dim=self.attention_embed_dim,
                num_heads=4
            )(x, self.body_part_indices)
        # 如果使用传统的全局注意力（向后兼容）
        elif self.use_attention:
            # 保留原有的注意力机制实现
            original_shape = x.shape
            if len(original_shape) == 1:
                x = x[None, None, :]
            elif len(original_shape) == 2:
                x = x[:, None, :]
            
            if x.shape[-1] != self.attention_embed_dim:
                x = nn.Dense(self.attention_embed_dim, 
                           kernel_init=orthogonal(np.sqrt(2)), 
                           bias_init=constant(0.0))(x)
            
            x = SelfAttention(self.attention_embed_dim)(x)
            x = jnp.mean(x, axis=-2)
        
        # 阶段2：多层网络 + 间隔通道注意力
        for i, dim_layer in enumerate(self.hidden_layer_dims):
            if self.use_residual and i > 0:
                x = ResidualBlock(dim_layer, self.activation)(x)
            else:
                x = nn.Dense(dim_layer, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
                x = self.activation_fn(x)
            
            # 在指定层后添加通道注意力
            if self.use_channel_attention and i in self.channel_attention_layers:
                x = ChannelAttention(reduction_ratio=self.channel_reduction_ratio)(x)

        # 输出层
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        x = self.output_activation_fn(x)

        return jnp.squeeze(x) if self.squeeze_output else x


class ImprovedActorCritic(nn.Module):
    """改进的Actor-Critic网络，支持多种注意力机制"""
    action_dim: Sequence[int]
    activation: str = "tanh"
    init_std: float = 1.0
    learnable_std: bool = True
    hidden_layer_dims: Sequence[int] = (1024, 512)
    actor_obs_ind: jnp.ndarray = None
    critic_obs_ind: jnp.ndarray = None
    use_residual: bool = False
    use_attention: bool = False
    attention_embed_dim: int = 64
    use_channel_attention: bool = False
    channel_attention_layers: Sequence[int] = (0, 2)
    channel_reduction_ratio: int = 16
    use_grouped_attention: bool = False
    body_part_indices: dict = None
    
    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)

    @nn.compact
    def __call__(self, x):
        x = RunningMeanStd()(x)

        # Build actor
        actor_x = x if self.actor_obs_ind is None else x[..., self.actor_obs_ind]
        actor_mean = ImprovedFullyConnectedNet(
            self.hidden_layer_dims, 
            self.action_dim, 
            self.activation,
            None, 
            False, 
            False,
            self.use_residual,
            self.use_attention,
            self.attention_embed_dim,
            self.use_channel_attention,
            self.channel_attention_layers,
            self.channel_reduction_ratio,
            self.use_grouped_attention,
            self.body_part_indices
        )(actor_x)
        
        actor_logtstd = self.param("log_std", nn.initializers.constant(jnp.log(self.init_std)),
                                   (self.action_dim,))
        if not self.learnable_std:
            actor_logtstd = jax.lax.stop_gradient(actor_logtstd)

        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        # Build critic
        critic_x = x if self.critic_obs_ind is None else x[..., self.critic_obs_ind]
        critic = ImprovedFullyConnectedNet(
            self.hidden_layer_dims, 
            1, 
            self.activation, 
            None, 
            False, 
            False,
            self.use_residual,
            self.use_attention,
            self.attention_embed_dim,
            self.use_channel_attention,
            self.channel_attention_layers,
            self.channel_reduction_ratio,
            self.use_grouped_attention,
            self.body_part_indices
        )(critic_x)

        return pi, jnp.squeeze(critic, axis=-1)






class FullyConnectedNet(nn.Module):
    hidden_layer_dims: Sequence[int]
    output_dim: int
    activation: str = "tanh"
    output_activation: str = None    # none means linear activation
    use_running_mean_stand: bool = True
    squeeze_output: bool = True
    use_residual: bool = False
    use_attention: bool = False
    attention_embed_dim: int = 64
    
    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)
        self.output_activation_fn = get_activation_fn(self.output_activation) \
            if self.output_activation is not None else lambda x: x

    @nn.compact
    def __call__(self, x):
        if self.use_running_mean_stand:
            x = RunningMeanStd()(x)

        # If using attention, reshape input for sequence processing
        if self.use_attention:
            # Assume input can be treated as a sequence
            original_shape = x.shape
            if len(original_shape) == 1:
                x = x[None, None, :]  # Add batch and sequence dims
            elif len(original_shape) == 2:
                x = x[None, :, :]  # Add batch dim
            elif len(original_shape) == 3:
                pass  # Already has batch, seq, feature dims
            else:
                # Flatten extra dimensions
                x = x.reshape(x.shape[0], -1, self.attention_embed_dim)
            
            # Project to attention embedding dimension if needed
            if x.shape[-1] != self.attention_embed_dim:
                x = nn.Dense(self.attention_embed_dim, 
                           kernel_init=orthogonal(np.sqrt(2)), 
                           bias_init=constant(0.0))(x)
            
            # Apply self-attention
            x = SelfAttention(self.attention_embed_dim)(x)
            
            # Global average pooling to reduce sequence dimension
            x = jnp.mean(x, axis=-2)
        
        # Build network with optional residual connections
        for i, dim_layer in enumerate(self.hidden_layer_dims):
            if self.use_residual and i > 0:  # Only add residual after first layer
                x = ResidualBlock(dim_layer, self.activation)(x)
            else:
                x = nn.Dense(dim_layer, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
                x = self.activation_fn(x)

        # Add last layer
        x = nn.Dense(self.output_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        x = self.output_activation_fn(x)

        return jnp.squeeze(x) if self.squeeze_output else x


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    init_std: float = 1.0
    learnable_std: bool = True
    hidden_layer_dims: Sequence[int] = (1024, 512)
    actor_obs_ind: jnp.ndarray = None
    critic_obs_ind: jnp.ndarray = None
    use_residual: bool = False
    use_attention: bool = False
    attention_embed_dim: int = 64
    
    def setup(self):
        self.activation_fn = get_activation_fn(self.activation)

    @nn.compact
    def __call__(self, x):
        x = RunningMeanStd()(x)

        # Build actor
        actor_x = x if self.actor_obs_ind is None else x[..., self.actor_obs_ind]
        actor_mean = FullyConnectedNet(
            self.hidden_layer_dims, 
            self.action_dim, 
            self.activation,
            None, 
            False, 
            False,
            self.use_residual,
            self.use_attention,
            self.attention_embed_dim
        )(actor_x)
        
        actor_logtstd = self.param("log_std", nn.initializers.constant(jnp.log(self.init_std)),
                                   (self.action_dim,))
        if not self.learnable_std:
            actor_logtstd = jax.lax.stop_gradient(actor_logtstd)

        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        # Build critic
        critic_x = x if self.critic_obs_ind is None else x[..., self.critic_obs_ind]
        critic = FullyConnectedNet(
            self.hidden_layer_dims, 
            1, 
            self.activation, 
            None, 
            False, 
            False,
            self.use_residual,
            self.use_attention,
            self.attention_embed_dim
        )(critic_x)

        return pi, jnp.squeeze(critic, axis=-1)


class RunningMeanStd(nn.Module):
    """Layer that maintains running mean and variance for input normalization."""

    @nn.compact
    def __call__(self, x):

        x = jnp.atleast_2d(x)

        # Initialize running mean, variance, and count
        mean = self.variable('run_stats', 'mean', lambda: jnp.zeros(x.shape[-1]))
        var = self.variable('run_stats', 'var', lambda: jnp.ones(x.shape[-1]))
        count = self.variable('run_stats', 'count', lambda: jnp.array(1e-6))

        # Compute batch mean and variance
        batch_mean = jnp.mean(x, axis=0)
        batch_var = jnp.var(x, axis=0) + 1e-6  # Add epsilon for numerical stability
        batch_count = x.shape[0]

        # Update counts
        updated_count = count.value + batch_count

        # Numerically stable mean and variance update
        delta = batch_mean - mean.value
        new_mean = mean.value + delta * batch_count / updated_count

        # Compute the new variance using Welford's method
        m_a = var.value * count.value
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * count.value * batch_count / updated_count
        new_var = M2 / updated_count

        # Normalize input
        normalized_x = (x - new_mean) / jnp.sqrt(new_var + 1e-8)

        # Update state variables
        mean.value = new_mean
        var.value = new_var
        count.value = updated_count

        return jnp.squeeze(normalized_x)