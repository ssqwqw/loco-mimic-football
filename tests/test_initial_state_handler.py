import numpy.random

from loco_mujoco.core.utils import mj_jntname2qposid

from test_conf import *

# set Jax-backend to CPU
jax.config.update('jax_platform_name', 'cpu')
print(f"Jax backend device: {jax.default_backend()} \n")


def _create_env(backend, init_state_type, init_state_params=None, trajectory=None):
    DEFAULTS = {"horizon": 1000, "gamma": 0.99, "n_envs": 1}

    mjx_env = DummyHumamoidEnv(enable_mjx=backend == "jax",
                               init_state_type=init_state_type,
                               init_state_params=init_state_params,
                               **DEFAULTS)

    if trajectory is not None:
        if backend == "numpy":
            trajectory.data = trajectory.data.to_numpy()

        mjx_env.load_trajectory(trajectory)

    return mjx_env

@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_DefaultInitialStateHandler(falling_trajectory, backend):
    seed = 0
    key = jax.random.PRNGKey(seed)

    mjx_env_1 = _create_env(backend, "DefaultInitialStateHandler")

    # create init_state_params
    qpos_init = np.zeros(18)
    qpos_init[:7] = np.array([1.5, 1.2, 0.3, 1., 0,  0., 0.])
    j_id = mj_jntname2qposid("abdomen_z", mjx_env_1._model)
    qpos_init[j_id] = 0.3

    qvel_init = 0.1 * np.ones(17)

    init_state_params = dict(qpos_init=qpos_init, qvel_init=qvel_init)

    mjx_env_2 = _create_env(backend, "DefaultInitialStateHandler", init_state_params=init_state_params)

    if backend == "numpy":
        state_0 = mjx_env_1.reset(key)
        state_1 = mjx_env_2.reset(key)

        print("\nstate_0", state_0)
        print("position: ", mjx_env_1._data.qpos[0:3])
        print("quaternion: ", mjx_env_1._data.qpos[3:7])
        print("quaternion[4]: ", mjx_env_1._data.qpos[6])
        state_0_test = np.array([ 1.293, 1., 0., 0., 0., # FreeJointPosNoXY
                                  0., # JointPos
                                  0., 0., 0., 0., 0., 0., # FreeJointVel
                                  0., # JointVel
                                  -0.02726929, 0.09849757, 0.828, # BodyPos
                                  0., 0., 0., 0., 0., 0. # BodyVel
                                ])

        assert np.allclose(state_0, state_0_test, atol=1e-7)

        state_1_test = np.array([0.3, 1., 0., 0., 0., # FreeJointPosNoXY
                                 0.3, # JointPos
                                 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, # FreeJointVel
                                 .1, # JointVel
                                 -0.02726929, 0.09849757, 0.828, # BodyPos
                                  0., 0., 0., 0., 0., 0. # BodyVel
                                 ])
        print("\nstate_1", state_1)
        assert np.allclose(state_1, state_1_test)

        assert np.allclose(mjx_env_2._data.qpos, qpos_init, atol=1e-7)
        assert np.allclose(mjx_env_2._data.qvel, qvel_init, atol=1e-7)
    else:
        state_0 = mjx_env_1.mjx_reset(key)
        state_1 = mjx_env_2.mjx_reset(key)

        print("\nstate_0", state_0.observation)
        state_0_test = jnp.array([1.293, 1., 0., 0., 0.,  # FreeJointPosNoXY
                                  0.,  # JointPos
                                  0., 0., 0., 0., 0., 0.,  # FreeJointVel
                                  0.,  # JointVel
                                  -0.02726929, 0.09849757, 0.828,  # BodyPos
                                  0., 0., 0., 0., 0., 0.  # BodyVel
                                 ])

        assert jnp.allclose(state_0.observation, state_0_test, atol=1e-7)

        state_1_test = jnp.array([0.3, 1., 0., 0., 0.,  # FreeJointPosNoXY
                                  0.3,  # JointPos
                                  0.1, 0.1, 0.1, 0.1, 0.1, 0.1,  # FreeJointVel
                                  .1,  # JointVel
                                  -0.02726929, 0.09849757, 0.828,  # BodyPos
                                  0., 0., 0., 0., 0., 0.  # BodyVel
                                  ])
        print("\nstate_1", state_1.observation)
        assert jnp.allclose(state_1.observation, state_1_test, atol=1e-7)

        assert jnp.allclose(state_1.data.qpos, qpos_init, atol=1e-7)
        assert jnp.allclose(state_1.data.qvel, qvel_init, atol=1e-7)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_TrajInitialStateHandler(falling_trajectory, backend, mock_random):
    mjx_env = _create_env(backend, "TrajInitialStateHandler", trajectory=falling_trajectory)

    seed = 0
    key = jax.random.PRNGKey(seed)
    numpy.random.seed(seed)

    if backend == "numpy":
        state_0 = mjx_env.reset(key)
        state_1 = mjx_env.reset(key)
        state_2 = mjx_env.reset(key)

        state_0_test = np.array([ 1.15832865e-01,  5.34948349e-01, -5.92454337e-02, -7.49823511e-01,
                                3.84818047e-01,  3.12941559e-02,  2.00834777e-03,  6.30782393e-04,
                                1.49963584e-04, -4.94559947e-03,  2.98579875e-03,  7.47354515e-03,
                                -1.63021516e-02, -1.72109634e-01,  2.61873245e-01,  7.79013932e-02,
                                -1.44799496e-03, -4.05410631e-03, -1.03898430e-02, -3.81383346e-03,
                                3.93864495e-04, -2.82701687e-04])
        state_1_test = np.array([ 1.15832865e-01,  5.34948349e-01, -5.92454337e-02, -7.49823511e-01,  
                                3.84818047e-01,  3.12941559e-02,  2.00834777e-03,  6.30782393e-04,  
                                1.49963584e-04, -4.94559947e-03,  2.98579875e-03,  7.47354515e-03,  
                                -1.63021516e-02, -1.72109634e-01,  2.61873245e-01,  7.79013932e-02,  
                                -1.44799496e-03, -4.05410631e-03, -1.03898430e-02, -3.81383346e-03,  
                                3.93864495e-04, -2.82701687e-04]
                                )
        state_2_test = np.array([ 1.15832865e-01,  5.34948349e-01, -5.92454337e-02, -7.49823511e-01,
                                3.84818047e-01,  3.12941559e-02,  2.00834777e-03,  6.30782393e-04,
                                1.49963584e-04, -4.94559947e-03,  2.98579875e-03,  7.47354515e-03,
                                -1.63021516e-02, -1.72109634e-01,  2.61873245e-01,  7.79013932e-02,
                                -1.44799496e-03, -4.05410631e-03, -1.03898430e-02, -3.81383346e-03,
                                3.93864495e-04, -2.82701687e-04])

        print("\nstate_0", state_0)
        assert np.allclose(state_0, state_0_test, atol=1e-7)

        print("\nstate_1", state_1)
        assert np.allclose(state_1, state_1_test, atol=1e-7)

        print("\nstate_2", state_2)
        assert np.allclose(state_2, state_2_test, atol=1e-7)
    else:
        state_0 = mjx_env.mjx_reset(key)
        key = jax.random.PRNGKey(seed+1)
        state_1 = mjx_env.mjx_reset(key)
        key = jax.random.PRNGKey(seed+2)
        state_2 = mjx_env.mjx_reset(key)

        state_0_test = np.array([ 1.15835026e-01,  5.34945846e-01, -5.92810139e-02, -7.49824584e-01,
                                3.84814054e-01,  3.11568417e-02,  1.99284335e-03,  5.12760831e-04,
                                2.53190839e-04, -4.26427508e-03,  1.89024326e-03,  3.65291978e-03,
                                -1.21903252e-02, -1.72136158e-01,  2.61878371e-01,  7.78965503e-02,
                                -1.47610775e-03, -4.04779986e-03, -1.03657534e-02, -3.78678367e-03,
                                4.08710970e-04, -2.78261869e-04])
        state_1_test = np.array([ 1.15835026e-01,  5.34945846e-01, -5.92810139e-02, -7.49824584e-01,  
                                3.84814054e-01,  3.11568417e-02,  1.99284335e-03,  5.12760831e-04,  
                                2.53190839e-04, -4.26427508e-03,  1.89024326e-03,  3.65291978e-03,  
                                -1.21903252e-02, -1.72136158e-01,  2.61878371e-01,  7.78965503e-02,  
                                -1.47610775e-03, -4.04779986e-03, -1.03657534e-02, -3.78678367e-03,  
                                4.08710970e-04, -2.78261869e-04])
        state_2_test = np.array([ 1.15835026e-01,  5.34945846e-01, -5.92810139e-02, -7.49824584e-01,
                                3.84814054e-01,  3.11568417e-02,  1.99284335e-03,  5.12760831e-04,
                                2.53190839e-04, -4.26427508e-03,  1.89024326e-03,  3.65291978e-03,
                                -1.21903252e-02, -1.72136158e-01,  2.61878371e-01,  7.78965503e-02,
                                -1.47610775e-03, -4.04779986e-03, -1.03657534e-02, -3.78678367e-03,
                                4.08710970e-04, -2.78261869e-04])

        print("\nstate_0", state_0.observation)
        assert np.allclose(state_0.observation, state_0_test, atol=1e-7)

        print("\nstate_1", state_1.observation)
        assert np.allclose(state_1.observation, state_1_test, atol=1e-7)

        print("\nstate_2", state_2.observation)
        assert np.allclose(state_2.observation, state_2_test, atol=1e-7)
