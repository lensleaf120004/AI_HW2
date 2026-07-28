import numpy as np


# 控制邊界
def fixed_marginal(value, lower_bound, upper_bound):

    if value < lower_bound:
        return lower_bound
    elif value > upper_bound:
        return upper_bound
    else:
        return value

def hill_climbing(loss_fn, start_points, field_size, steps=30):
    """
    Fixed 8-direction hill climbing with padding
      1.以 8 方向鄰居搜尋；若沒有更佳鄰居則提前停止
      2.若提前停止，剩餘步數以「原地不動」補齊，總長度為 steps+1（含起點）
    """

    # 邊界
    lower_bound = float(field_size[0])
    upper_bound = float(field_size[1])

    # 座標上的八個方向
    directions = []
    directions.append(( 1,  0))
    directions.append((-1,  0))
    directions.append(( 0,  1))
    directions.append(( 0, -1))
    directions.append(( 1,  1))
    directions.append(( 1, -1))
    directions.append((-1,  1))
    directions.append((-1, -1))

    # 步長 : 0.2 -> 2.5
    step_size = float(0.5)

    # 輸入起點 => 轉可修改的陣列
    starts_array = np.array(start_points, dtype=float).copy()
    num_ships = int(len(starts_array))

    
    paths = []
    idx_init = 0
    while idx_init < num_ships:
        paths.append([])
        idx_init += 1

    # 執行 Hill Climbing(執行一次->本題只有一艘船)
    ship_index = 0
    while ship_index < num_ships:
        # 讀取起點
        start_x = float(starts_array[ship_index][0])
        start_y = float(starts_array[ship_index][1])

        x = start_x
        y = start_y

        # 計算起點GPA
        current_gpa = float(loss_fn(x, y))

        # 個人最佳（初始位置和GPA為起點座標及其GPA）
        best_x = x
        best_y = y
        best_gpa = current_gpa

        # 路徑(包含起點)
        path = []
        path.append((x, y, current_gpa, best_x, best_y, best_gpa))

        # 紀錄已移動的步數
        moves_used = 0

        # main loop
        while moves_used < steps:
            # 搜尋8方向鄰居中GPA最好
            neighbor_best_gpa = float("-inf")
            neighbor_best_x = x
            neighbor_best_y = y

            dir_idx = 0
            while dir_idx < len(directions):
                dx = float(directions[dir_idx][0])
                dy = float(directions[dir_idx][1])

                cand_x = x + dx * step_size
                cand_y = y + dy * step_size

                # 邊界限制固定
                cand_x = fixed_marginal(cand_x, lower_bound, upper_bound)
                cand_y = fixed_marginal(cand_y, lower_bound, upper_bound)

                cand_gpa = float(loss_fn(cand_x, cand_y))

                if cand_gpa > neighbor_best_gpa:
                    neighbor_best_gpa = cand_gpa
                    neighbor_best_x = cand_x
                    neighbor_best_y = cand_y

                dir_idx += 1

            # 若最佳鄰居的GPA優於目前位置的GPA → 前進；否則提前停止
            if neighbor_best_gpa > current_gpa:
                x = neighbor_best_x
                y = neighbor_best_y
                current_gpa = neighbor_best_gpa
                moves_used = moves_used + 1

                # 更新個人最佳GPA
                if current_gpa > best_gpa:
                    best_x = x
                    best_y = y
                    best_gpa = current_gpa

                # 記錄路徑
                path.append((x, y, current_gpa, best_x, best_y, best_gpa))
            else:
                # 無更佳鄰居就會先提前停止
                break

        # 針對提前停止的情況(小於30步)：以「原地不動」補齊到 steps
        if moves_used < steps:
            last_tuple = path[-1]  # (x, y, gpa, best_x, best_y, best_gpa)
            remain = int(steps - moves_used)
            k = 0
            while k < remain:
                path.append(last_tuple)
                k += 1
        # ====== 補齊結束 ======

        # 放入總 paths
        paths[ship_index] = path

        ship_index += 1

    return paths
