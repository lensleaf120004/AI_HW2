import numpy as np


# 控制邊界
def fixed_marginal(value, lower_bound, upper_bound):
    
    if value < lower_bound:
        return lower_bound
    elif value > upper_bound:
        return upper_bound
    else:
        return value

def simulated_annealing(loss_fn, start_points, field_size, steps=30):
    """
    Simulated Annealing
      1.每一步從 {-step_size, 0, +step_size}^2（排除(0,0)）隨機挑鄰居
      2.依 delta 與溫度 temp 決定是否接受較差解（~ exp(delta / temp)）
      3.回傳路徑：每步 (x, y, gpa, best_x, best_y, best_gpa, temp)
    """

    # 邊界
    lower_bound = float(field_size[0])
    upper_bound = float(field_size[1])

    
    starts_array = np.array(start_points, dtype=float).copy()
    num_ships = int(len(starts_array))

    
    paths = []
    idx_init = 0
    while idx_init < num_ships:
        paths.append([])
        idx_init += 1

    # 執行 simulated annealing(執行一次->本題只有一艘船)
    ship_index = 0
    while ship_index < num_ships:
        # 初始狀態
        start_x = float(starts_array[ship_index][0])
        start_y = float(starts_array[ship_index][1])

        x = start_x
        y = start_y

        current_gpa = float(loss_fn(x, y))

        best_x = x
        best_y = y
        best_gpa = current_gpa

        # simualted annealing 溫度參數
        temp_start = float(2.0)
        temp_decay = float(0.91)   # 每步溫度衰退率
        temp_min   = float(1e-8)  # 溫度下限
        step_size  = float(0.5)   # 步長 : 0.2 -> 2.5

        temp = float(temp_start)

        # 路徑(包含起點)
        path = []
        path.append((x, y, current_gpa, best_x, best_y, best_gpa, temp))

        # main loop
        step_counter = 0
        while step_counter < int(steps):
            # 1.隨機挑一鄰居方向（排除 (0,0)）
            # choices = [-step_size, 0.0, +step_size]
            
            valid_neighbor = False
            dx = 0.0
            dy = 0.0

            # 直到挑到非 (0,0)
            while not valid_neighbor:
                r1 = np.random.randint(0, 3)  # 0,1,2 -> -step, 0, +step
                r2 = np.random.randint(0, 3)
                if r1 == 0:
                    dx = -step_size
                elif r1 == 1:
                    dx = 0.0
                else:
                    dx = step_size

                if r2 == 0:
                    dy = -step_size
                elif r2 == 1:
                    dy = 0.0
                else:
                    dy = step_size

                if not (dx == 0.0 and dy == 0.0):
                    valid_neighbor = True

            # 2.計算候選點與其 GPA（限制邊界）
            nx = fixed_marginal(x + dx, lower_bound, upper_bound)
            ny = fixed_marginal(y + dy, lower_bound, upper_bound)
            next_gpa = float(loss_fn(nx, ny))

            # 3.差值
            delta = float(next_gpa - current_gpa)

            # 4.變好一定採用；變差以 exp(delta / temp) 機率採用
            accept_move = False
            if delta > 0.0:
                accept_move = True
            else:
                # 避免溫度很小情況
                t_eff = temp
                if t_eff < temp_min:
                    t_eff = temp_min
                prob = float(np.exp(delta / t_eff))  # 0~1 的採用率
                rand_v = float(np.random.rand())
                if rand_v < prob:
                    accept_move = True

            if accept_move:
                x = float(nx)
                y = float(ny)
                current_gpa = float(next_gpa)

            # 5.更新最佳
            if current_gpa > best_gpa:
                best_x = x
                best_y = y
                best_gpa = current_gpa

            # 6.溫度衰退
            temp = float(temp * temp_decay)
            if temp < temp_min:
                temp = temp_min

            # 紀錄
            path.append((x, y, current_gpa, best_x, best_y, best_gpa, temp))

            step_counter = step_counter + 1

        # 放入總 paths
        paths[ship_index] = path
        ship_index = ship_index + 1

    return paths
