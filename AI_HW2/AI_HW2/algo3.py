import numpy as np


# 控制邊界
def fixed_marginal(value, lower_bound, upper_bound):
    # C-style: explicit, verbose clamp
    if value < lower_bound:
        return lower_bound
    elif value > upper_bound:
        return upper_bound
    else:
        return value

def ultimate_algorithm(loss_fn, start_points, field_size, steps=30):
    """
    Improved Simulated Annealing (單軸移動 + 每步結束移動共享最佳GPA的位置資訊(聚攏) + padding:小於30步內的補齊步數)
      1.多艘船並行，每艘獨立 SA 鏈
      2.沿 X 或 Y 單軸移動（步幅隨溫度縮放->步幅調整適應）
      3.偶爾大跳：小機率大尺度探索
      4.回溫：卡住時稍微升溫
      5.共享最佳位置資訊：定期朝全域最佳GPA的候選位置微調整
      6.padding：提前停止時，會把剩餘步數補滿（原地不動）

    輸出每步: (x, y, gpa, best_x, best_y, best_gpa)
    """

    # 邊界
    lower_bound = float(field_size[0])
    upper_bound = float(field_size[1])

    # 溫度控制參數
    TEMP_START  = 3.0
    TEMP_DECAY  = 0.94
    TEMP_MIN    = 1e-8

    PATIENCE    = 6
    REHEAT      = 1.2

    SPAN_RANGE        = upper_bound - lower_bound
    BASE_STEP   = 0.30 * SPAN_RANGE
    MIN_STEP    = 0.02 * SPAN_RANGE

    # 偶爾大跳躍移動避免卡住(跳躍參數)
    P_BIG_JUMP  = 0.06
    BIG_JUMP_SD = 0.55 * SPAN_RANGE

    # 開啟共享機制(聚攏時參數)
    BEST_SHARE = True 
    SHARE_EVERY   = 4
    PULL_RATIO    = 0.35
    

    # 初始設定
    ships_array = np.array(start_points, dtype=float).copy()
    num_ships = int(len(ships_array))

    # 路徑
    paths = []
    for _idx in range(num_ships):
        paths.append([])

    # 溫度、停滯計次
    temperatures = np.full(num_ships, TEMP_START, dtype=float)
    no_improve_counter = np.zeros(num_ships, dtype=int)

    # 當前位置與GPA
    current_xy = ships_array.astype(float)
    current_gpa = np.zeros(num_ships, dtype=float)
    for i in range(num_ships):
        x0 = float(current_xy[i][0])
        y0 = float(current_xy[i][1])
        g0 = float(loss_fn(x0, y0))
        current_gpa[i] = g0

    # 個人最佳
    best_xy = current_xy.copy()
    best_gpa = current_gpa.copy()

    # 路徑(包含起點)
    for i in range(num_ships):
        init_x = float(current_xy[i][0])
        init_y = float(current_xy[i][1])
        init_g = float(current_gpa[i])
        best_x = float(best_xy[i][0])
        best_y = float(best_xy[i][1])
        best_g = float(best_gpa[i])
        paths[i].append((init_x, init_y, init_g, best_x, best_y, best_g))

    # outer main loop : 在限制步數內
    t = 1
    while t <= steps:
        # 計算全域最佳（best_share 用）
        global_best_index = int(np.argmax(best_gpa))
        global_best_x = float(best_xy[global_best_index][0])
        global_best_y = float(best_xy[global_best_index][1])

        # inner 2nd loop : 逐個船隻更新
        i = 0
        while i < num_ships:
            # 若這艘船已經被 padding 補滿（之前提前停止）=> 略過
            if len(paths[i]) >= (steps + 1):
                i += 1
                continue

            x = float(current_xy[i][0])
            y = float(current_xy[i][1])
            g_now = float(current_gpa[i])
            temp_now = float(temperatures[i])
            if temp_now < TEMP_MIN:
                temp_now = TEMP_MIN

            # 1.步幅調整、適應（和溫度成比例，但確保不會小於 MIN_STEP）
            alpha = float(temperatures[i] / TEMP_START)
            step_len = float(alpha * BASE_STEP)
            if step_len < MIN_STEP:
                step_len = MIN_STEP

            ## 產生單軸向的鄰居（只動X或Y軸）
            rand_val_axis = np.random.rand()
            rand_val_sign = np.random.rand()
            if rand_val_axis < 0.5:
                # move in X only
                if rand_val_sign < 0.5:
                    nx = x + step_len
                else:
                    nx = x - step_len
                ny = y
            else:
                # move in Y only
                nx = x
                if rand_val_sign < 0.5:
                    ny = y + step_len
                else:
                    ny = y - step_len

            ## 偶爾的大跳
            rand_jump = np.random.rand()
            if rand_jump < P_BIG_JUMP:
                nx = nx + float(np.random.randn() * BIG_JUMP_SD)
                ny = ny + float(np.random.randn() * BIG_JUMP_SD)

            # 邊界限制固定
            nx = fixed_marginal(nx, lower_bound, upper_bound)
            ny = fixed_marginal(ny, lower_bound, upper_bound)

            # 2.Best_Share：在分享步時把「朝最優靠近」當第二候選，二選一
            if BEST_SHARE and ((t % SHARE_EVERY) == 0):
                ex = x + PULL_RATIO * (global_best_x - x)
                ey = y + PULL_RATIO * (global_best_y - y)
                ex = fixed_marginal(float(ex), lower_bound, upper_bound)
                ey = fixed_marginal(float(ey), lower_bound, upper_bound)

                g_axis = float(loss_fn(nx, ny))
                g_best = float(loss_fn(ex, ey))
                if g_best > g_axis:
                    cand_x = float(ex)
                    cand_y = float(ey)
                    g_next = float(g_best)
                else:
                    cand_x = float(nx)
                    cand_y = float(ny)
                    g_next = float(g_axis)
            else:
                cand_x = float(nx)
                cand_y = float(ny)
                g_next = float(loss_fn(cand_x, cand_y))

            # 3.採用準則
            delta = float(g_next - g_now)
            accept_move = False
            if delta > 0.0:
                accept_move = True
            else:
                # np.exp(delta / temp_now) 可能非常小，這裡不特別截斷
                prob = float(np.exp(delta / temp_now))
                rand_accept = float(np.random.rand())
                if rand_accept < prob:
                    accept_move = True

            if accept_move:
                x = cand_x
                y = cand_y
                g_now = g_next
            else:
                # 若完全沒有改善、且接近冷卻且長時間無改進的情況 => 提前停止，後續補上 padding
                if (temperatures[i] < 0.05) and (no_improve_counter[i] >= PATIENCE):
                    remaining = int(steps - t + 1)
                    last_best_x = float(best_xy[i][0])
                    last_best_y = float(best_xy[i][1])
                    last_best_g = float(best_gpa[i])
                    last_state_tuple = (x, y, g_now, last_best_x, last_best_y, last_best_g)
                    k = 0
                    while k < remaining:
                        paths[i].append(last_state_tuple)
                        k += 1
                    current_xy[i][0] = x
                    current_xy[i][1] = y
                    current_gpa[i] = g_now
                    i += 1
                    continue

            # 4.更新個人最佳與停滯計次
            if g_now > best_gpa[i]:
                best_gpa[i] = g_now
                best_xy[i][0] = x
                best_xy[i][1] = y
                no_improve_counter[i] = 0
            else:
                no_improve_counter[i] = int(no_improve_counter[i] + 1)

            # 5.降溫 + 回溫
            new_temp = float(temp_now * TEMP_DECAY)
            if new_temp < TEMP_MIN:
                new_temp = TEMP_MIN
            temperatures[i] = new_temp

            if no_improve_counter[i] >= PATIENCE:
                reheated = float(temperatures[i] * REHEAT)
                if reheated > TEMP_START:
                    reheated = TEMP_START
                temperatures[i] = reheated
                no_improve_counter[i] = 0

            # 6.記錄狀態
            current_xy[i][0] = x
            current_xy[i][1] = y
            current_gpa[i] = g_now

            record_best_x = float(best_xy[i][0])
            record_best_y = float(best_xy[i][1])
            record_best_g = float(best_gpa[i])
            paths[i].append((x, y, g_now, record_best_x, record_best_y, record_best_g))

            i += 1  # 下一艘船(inner)

        t += 1  # 下一個時間步(outer)

    # 確保每艘船路徑長度一致（起點 + steps）
    i = 0
    while i < num_ships:
        required_len = int(steps + 1)
        current_len = int(len(paths[i]))
        if current_len < required_len:
            last_state = paths[i][-1]
            count_to_fill = int(required_len - current_len)
            k = 0
            while k < count_to_fill:
                paths[i].append(last_state)
                k += 1
        i += 1

    return paths
