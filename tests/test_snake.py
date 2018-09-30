import Snake
snake = Snake.Snake((1, 1))

def test_init_pos():
    assert snake.get_pos() == (1, 1)

def test_init_len():
    assert snake.length == 3
    assert len(snake.elements) == 3

def test_current_dir():
    assert snake.current_dir == 'south'

def test_snake_elements():
    assert snake.elements[1] == (1, 0)


def test_move_from_south_left():
    snake.move(0)
    assert snake.current_dir == 'east'
    assert snake.get_pos() == (2, 1)

def test_move_from_east_straight():
    snake.move(1)
    assert snake.current_dir == 'east'
    assert snake.get_pos() == (3, 1)

def test_move_from_east_right():
    snake.move(2)
    assert snake.current_dir == 'south'
    assert snake.get_pos() == (3, 2)

def test_move_from_south_right():
    snake.move(2)
    assert snake.current_dir == 'west'
    assert snake.get_pos() == (2, 2)

def test_move_head_attached():
    snake.move(0)
    assert snake.current_dir == 'south'
    assert snake.get_pos() == (2, 3)
    assert snake.elements[0] == (2, 3)


def test_move_from_south_invalid():
    try:
        snake.move(3)
    except ValueError:
        assert True




