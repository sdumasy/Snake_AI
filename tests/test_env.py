import Env
import numpy as np
import Snake

env = Env.Env()

def test_env_init():
    assert env.block_size == 20
    assert env.res == [440, 440]
    assert env.field_size == 22

def test_init_matrix():
    reset_clean_map_snake_on_pos((9,9))
    assert env.matrix[0, 0] == 1
    assert env.matrix[0, 5] == 1
    assert env.matrix[7, 0] == 1
    assert env.matrix[21, 0] == 1
    assert env.matrix[3, 2] == 0
    assert env.matrix[21, 21] == 1
    assert env.matrix[6, 6] == 0

def test_matrix_to_game_dim():
    pos = 0
    pos1 = 6
    pos2 = 11
    assert env.matrix_to_game_dims(pos) == pos * 20 + 10
    assert env.matrix_to_game_dims(pos1) == pos1 * 20 + 10
    assert env.matrix_to_game_dims(pos2) == pos2 * 20 + 10

def reset_clean_map_snake_on_pos(pos, pos2 = (15,15)):
    env.reset_env()
    env.create_map()
    env.create_snake(pos)
    env.create_bug(pos2)

def test_take_action():
    reset_clean_map_snake_on_pos((1,5))
    env.take_action(1)
    assert env.take_action(2) == (-10, True)

def test_take_action_2():
    reset_clean_map_snake_on_pos((5,5))
    env.take_action(1)
    assert env.take_action(0) == (0, False)

def test_take_action_3():
    reset_clean_map_snake_on_pos((19,20))
    env.take_action(0)
    assert env.take_action(1) == (-10, True)

def test_take_action_4():
    reset_clean_map_snake_on_pos((19,20))
    env.take_action(0)
    assert env.take_action(2) == (-10, True)

def test_take_action_5():
    reset_clean_map_snake_on_pos((19,20))
    env.take_action(0)
    assert env.take_action(1) == (-10, True)

def test_take_action_6():
    reset_clean_map_snake_on_pos((19,20))
    env.take_action(0)
    assert env.take_action(1) == (-10, True)

def test_draw_snake():
    reset_clean_map_snake_on_pos((3,3))
    assert env.matrix[3, 3] == 3
    assert env.matrix[2, 3] == 2
    assert env.matrix[1, 3] == 2

def test_draw_snake2():
    try:
        reset_clean_map_snake_on_pos((0,0))
    except ValueError:
        assert True

def test_check_matrix():
    reset_clean_map_snake_on_pos((3,3))
    assert env.check_matrix((22,22)) == 1
    assert env.check_matrix((21,21)) == 1
    assert env.check_matrix((5,21)) == 1
    assert env.check_matrix((2,2)) == 0
    assert env.check_matrix((3,3)) == 3
    assert env.check_matrix((3,2)) == 2
    assert env.check_matrix((3,1)) == 2
    assert env.check_matrix((3,0)) == 1
    assert env.check_matrix((1,21)) == 1

def test_dead():
    reset_clean_map_snake_on_pos((1,3))
    assert env.check_dead() == False
    env.take_action(2)
    assert env.check_dead() == True

def test_dead2():
    reset_clean_map_snake_on_pos((1,20))
    assert env.check_dead() == False
    env.take_action(2)
    assert env.check_dead() == True

def test_check_bug():
    env.reset_env()
    env.create_map()
    env.create_bug((5,5))
    env.create_snake((5,4))
    env.take_action(1)
    assert len(np.argwhere(env.matrix == 5)) == 1
    assert env.reward == 10

def test_check_bug2():
    env.reset_env()
    env.create_map()
    env.create_bug((5,5))
    env.create_snake((5,3))
    env.take_action(1)
    env.take_action(1)
    assert env.reward == 10
    assert len(np.argwhere(env.matrix == 5)) == 1

def test_check_bug3():
    for i in range(100):
        env.reset_env()
        assert len(np.argwhere(env.matrix == 5)) == 1
        assert len(np.argwhere(env.matrix == 3)) == 1
        assert len(np.argwhere(env.matrix == 2)) == 2

def checkEqual(L1, L2):
    return len(L1) == len(L2) and sorted(L1) == sorted(L2)

def test_get_features():
    reset_clean_map_snake_on_pos((1,3), (20, 20))
    env.take_action(1)
    assert checkEqual(env.get_features(),  [0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0])
    env.take_action(1)

def test_get_features2():
    reset_clean_map_snake_on_pos((20,18), (5, 20))
    env.take_action(1)
    assert checkEqual(env.get_features(),  [0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1])
    env.take_action(1)
    assert checkEqual(env.get_features(),  [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1])
    env.take_action(2)
    assert checkEqual(env.get_features(),  [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1])
    env.take_action(2)
    assert checkEqual(env.get_features(),  [1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1])

def test_get_features3():
    reset_clean_map_snake_on_pos((20,18), (20, 20))
    env.take_action(1)
    env.take_action(1)
    assert checkEqual(env.get_features(),  [0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 1])
    assert env.score == 1

def test_get_features4():
    print("LAST TEST")
    reset_clean_map_snake_on_pos((10,10), (20, 20))
    env.take_action(1)
    env.take_action(1)
    assert checkEqual(env.get_features(),  [0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0])
