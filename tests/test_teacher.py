import torch

from somahouse.brain import HouseBrain
from somahouse.sim import baseline, observe, scenarios, transition
from somahouse.teacher import SEEDS, STRESS, TEST, VAL, Teacher, reward_loss


def test_new_study_splits_do_not_reuse_original_final_data():
    from somahouse import study
    splits = [set(VAL), set(TEST), set(STRESS), set(study.VAL), set(study.TEST), set(study.STRESS)]
    assert not set(SEEDS) & set(study.SEEDS)
    for i, left in enumerate(splits):
        assert all(not left & right for right in splits[i + 1:])
    assert min(VAL) > 5_999_999 and max(VAL) < min(TEST) and max(TEST) < min(STRESS)


def test_original_teacher_is_exact_and_new_teacher_respects_no_future_information():
    env = scenarios([5_100_001, 5_100_002], 2, training=True)
    obs = observe(env.initial, env, 0)
    assert torch.equal(Teacher({"original": True})(obs), baseline(obs, "balance"))
    teacher = Teacher({"gain": 1.3, "offset": .4, "co2_target": 950., "vent_gain": 1.2})
    initial = teacher(obs)
    env.outside[1:] += 20; env.people[1:] += 4
    assert torch.equal(initial, teacher(observe(env.initial, env, 0)))
    assert initial.shape == (2, 9) and torch.isfinite(initial).all()


def test_reward_training_changes_fly_recurrence_without_teacher_at_inference():
    torch.set_num_threads(2)
    env = scenarios([5_200_001, 5_200_002], 2, training=True)
    brain = HouseBrain(32, 107, copies=2)
    state, metrics = transition(env.initial, brain(observe(env.initial, env, 0)), env, 0)
    loss = reward_loss(metrics, state, env, 0)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(brain.recurrent.grad).all()
    assert ((brain.recurrent.grad != 0) & brain.edge_mask).any()
