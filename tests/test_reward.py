from math import isclose

from test_conf import *

# set Jax-backend to CPU
jax.config.update('jax_platform_name', 'cpu')
print(f"Jax backend device: {jax.default_backend()} \n")


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_NoReward(standing_trajectory, falling_trajectory, backend):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend, horizon=100,
                                             reward_type="NoReward")

    if backend == "numpy":
        assert len(transitions.rewards) == 99
        assert np.all(transitions.rewards == 0)
    else:
        assert len(transitions.rewards) == 99
        assert jnp.all(transitions.rewards == 0)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_TargetXVelocityReward(standing_trajectory, falling_trajectory, backend, mock_random):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    reward_params = dict(target_velocity=1.0)
    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend, horizon=100,
                                             reward_type="TargetXVelocityReward",
                                             reward_params=reward_params)

    if backend == "numpy":
        assert len(transitions.rewards) == 99

        reward_sum = transitions.rewards.sum()
        print("\nreward_sum: {0:.15f}".format(reward_sum))
        assert isclose(reward_sum, 18.903825759887695, abs_tol=1e-7)

        reward_42 = transitions.rewards[42]
        print("reward_42: {0:.15f}".format(reward_42))
        assert isclose(reward_42, 0.137758985161781, abs_tol=1e-7)
    else:
        assert len(transitions.rewards) == 99

        reward_sum = transitions.rewards.sum()
        print("\nreward_sum: {0:.15f}".format(reward_sum))
        assert jnp.isclose(reward_sum, 18.903825759887695, atol=1e-7)

        reward_42 = transitions.rewards[42]
        print("reward_42: {0:.15f}".format(reward_42))
        assert jnp.isclose(reward_42, 0.137758985161781, atol=1e-7)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_TargetVelocityGoalReward(standing_trajectory, falling_trajectory, backend, mock_random):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend, horizon=100,
                                             goal_type="GoalRandomRootVelocity",
                                             reward_type="TargetVelocityGoalReward")

    if backend == "numpy":
        assert len(transitions.rewards) == 99

        reward_sum = transitions.rewards.sum()
        print("\nreward_sum: {0:.15f}".format(reward_sum))
        assert isclose(reward_sum, 40.183032989501953, abs_tol=1e-7)

        reward_42 = transitions.rewards[42]
        print("reward_42: {0:.15f}".format(reward_42))
        assert isclose(reward_42, 0.805188059806824, abs_tol=1e-7)
    else:
        assert len(transitions.rewards) == 99

        reward_sum = transitions.rewards.sum()
        print("\nreward_sum: {0:.15f}".format(reward_sum))
        assert jnp.isclose(reward_sum, 40.183040618896484, atol=1e-7)

        reward_42 = transitions.rewards[42]
        print("reward_42: {0:.15f}".format(reward_42))
        assert jnp.isclose(reward_42, 0.805188119411469, atol=1e-7)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_LocomotionReward(standing_trajectory, falling_trajectory, backend, mock_random):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    reward_params = dict(joint_position_limit_coeff=1.0)
    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend, horizon=100,
                                             goal_type="GoalRandomRootVelocity",
                                             reward_type="LocomotionReward",
                                             reward_params=reward_params)

    if backend == "numpy":
        assert len(transitions.rewards) == 99

        reward_sum = transitions.rewards.sum()
        print("\nreward_sum: {0:.15f}".format(reward_sum))
        assert isclose(reward_sum, 17.714370727539062, abs_tol=1e-7)

        reward_42 = transitions.rewards[42]
        print("reward_42: {0:.15f}".format(reward_42))
        assert isclose(reward_42, 0., abs_tol=1e-7)
    else:
        # todo: fix this once the Locomotion reward is updated
        pass
        # assert len(transitions.rewards) == 99
        #
        # reward_sum = transitions.rewards.sum()
        # print("\nreward_sum: {0:.15f}".format(reward_sum))
        # #assert isclose(reward_sum, 0.371385663747787)
        #
        # reward_42 = transitions.rewards[42]
        # print("reward_42: {0:.15f}".format(reward_42))
        # #assert isclose(reward_42, 0.)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_TargetVelocityTrajReward(standing_trajectory, falling_trajectory, backend, mock_random):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend, horizon=100,
                                             reward_type="TargetVelocityTrajReward")

    if backend == "numpy":
        assert len(transitions.rewards) == 99

        reward_sum = transitions.rewards.sum()
        print("\nreward_sum: {0:.15f}".format(reward_sum))
        assert isclose(reward_sum, 41.132110595703125, abs_tol=1e-7)

        reward_42 = transitions.rewards[42]
        print("reward_42: {0:.15f}".format(reward_42))
        assert isclose(reward_42, 0.417381376028061, abs_tol=1e-7)
    else:
        assert len(transitions.rewards) == 99

        reward_sum = transitions.rewards.sum()
        print("\nreward_sum: {0:.15f}".format(reward_sum))
        assert jnp.isclose(reward_sum, 41.132110595703125, atol=1e-7)

        reward_42 = transitions.rewards[42]
        print("reward_42: {0:.15f}".format(reward_42))
        assert jnp.isclose(reward_42, 0.417381376028061, atol=1e-7)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_MimicReward(standing_trajectory, falling_trajectory, backend, mock_random):
    expert_traj: Trajectory = standing_trajectory
    nominal_traj: Trajectory = falling_trajectory

    transitions = generate_test_trajectories(expert_traj, nominal_traj, backend, horizon=100,
                                             reward_type="MimicReward")

    if backend == "numpy":
        assert len(transitions.rewards) == 99

        reward_sum = transitions.rewards.sum()
        print("\nreward_sum: {0:.15f}".format(reward_sum))
        assert isclose(reward_sum, 35.612613677978516, abs_tol=1e-7)

        reward_42 = transitions.rewards[42]
        print("reward_42: {0:.15f}".format(reward_42))
        assert isclose(reward_42, 0.498908072710037, abs_tol=1e-7)
    else:
        assert len(transitions.rewards) == 99

        reward_sum = transitions.rewards.sum()
        print("\nreward_sum: {0:.15f}".format(reward_sum))
        assert jnp.isclose(reward_sum, 35.612613677978516, atol=1e-7)

        reward_42 = transitions.rewards[42]
        print("reward_42: {0:.15f}".format(reward_42))
        assert jnp.isclose(reward_42, 0.498908072710037, atol=1e-7)
