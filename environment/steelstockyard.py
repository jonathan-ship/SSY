import numpy as np
import random
import pygame
import time
import environment.plate as plate

random.seed = 42


# 강재 적치 위치 결정 환경
class Locating(object):  # 생성자에서 파일의 수, 최대 높이 등을 입력
    def __init__(self, num_pile=4, max_stack=4, inbound_plates=None, observe_inbounds=False, display_env=False):
        self.action_space = num_pile  # 가능한 action 수는 파일의 수로 설정
        self.max_stack = max_stack  # 한 파일에 적치 가능한 강재의 수
        self.empty = -1  # 빈 공간의 상태 표현 값
        self.stage = 0
        self.current_date = 0
        self.plates = [[] for _ in range(num_pile)]  # 각 파일을 빈 리스트로 초기화
        self.n_features = max_stack * num_pile
        self.observe_inbounds = observe_inbounds
        if inbound_plates:
            self.inbound_plates = inbound_plates
            self.inbound_clone = self.inbound_plates[:]
        else:
            self.inbound_plates = plate.generate_schedule()
            self.inbound_clone = self.inbound_plates[:]
        if display_env:  # 환경을 게임엔진으로 가시화하는 용도. 학습용시에는 사용하지 않음
            display = LocatingDisplay(self, num_pile, max_stack, 2)
            display.game_loop_from_space()

    def step(self, action):
        done = False
        inbound = self.inbound_plates.pop(0)  # 입고 강재 리스트 가장 위에서부터 강재를 하나씩 입고
        if len(self.plates[action]) == self.max_stack:  # 적치 강재가 최대 높이를 초과하면 실패로 간주
            done = True
            reward = -1.0
        else:
            self.plates[action].append(inbound)  # action 에 따라서 강재를 적치
            reward = self._calculate_reward(action)  # 해당 action 에 대한 보상을 계산
            self.stage += 1
        if len(self.inbound_plates) == 0:
            done = True
        elif self.inbound_plates[0].inbound != self.current_date:
            self.current_date = self.inbound_plates[0].inbound
            self._export_plates()
        next_state = self._get_state()  # 쌓인 강재들 리스트에서 state 를 계산
        if done:
            next_state = self._export_all_plates()
        return next_state, reward, done

    def reset(self, episode=4, hold=True):
        if not hold:
            self.inbound_plates = plate.generate_schedule()
            self.inbound_clone = self.inbound_plates[:]
        else:
            self.inbound_plates = self.inbound_clone[(episode-1) % len(self.inbound_clone)][:]
            random.shuffle(self.inbound_plates)
        self.plates = [[] for _ in range(self.action_space)]
        self.current_date = min(self.inbound_plates, key=lambda x: x.inbound).inbound
        self.stage = 0
        return self._get_state()

    def _calculate_reward(self, action):
        pile = self.plates[action]
        max_move = 0
        if len(pile) == 1:
            return 0
        for i, plate in enumerate(pile[:-1]):
            move = 0
            if i + max_move > len(pile):
                break
            for upper in pile[i + 1:]:
                if plate.outbound < upper.outbound:  # 하단의 강재 적치 기간이 짧은 경우 예상 크레인 횟수 증가
                    move += 1
            if move > max_move:  # 파일 내의 강재들 중 반출 시 예상 크레인 사용 횟수가 최대인 강재를 기준으로 보상 계산
                max_move = move
        reward = 2  # 예상 크레인 사용 횟수가 0인 경우 최대인 2의 보상
        if max_move != 0:
            reward = 1 / max_move  # 예상 크레인 사용 횟수의 역수로 보상 계산
        return reward

    def _get_state(self):
        if self.observe_inbounds:
            state = np.full([self.max_stack, self.action_space + 1], self.empty)
            daily_plate = [plate for plate in self.inbound_plates[:self.max_stack] if plate.inbound == self.current_date]
            new_plates = [daily_plate[::-1]] + self.plates[:]
        else:
            state = np.full([self.max_stack, self.action_space], self.empty)
            new_plates = self.plates[:]
        for i, pile in enumerate(new_plates):
            for j, plate in enumerate(pile):
                state[j, i] = plate.outbound - self.current_date
        state = np.flipud(state).flatten()
        return state

    def _export_plates(self):
        for pile in self.plates:
            outbounds = []
            for i, plate in enumerate(pile):
                if plate.outbound <= self.current_date:
                    outbounds.append(i)
            for index in outbounds[::-1]:
                del pile[index]

    def _export_all_plates(self):
        next_states = []
        while True:
            next_outbound_date = min(sum(self.plates, []), key=lambda x: x.outbound).outbound
            if next_outbound_date != self.current_date:
                self.current_date = next_outbound_date
                self._export_plates()
                next_state = self._get_state()
                next_states.append(next_state)
            if not sum(self.plates, []):
                break
        return next_states


# 환경을 가시화하는 용도, 사람이 action 을 입력해야하므로 학습시에는 실행하지 않음
class LocatingDisplay(object):
    white = (255, 255, 255)
    black = (0, 0, 0)

    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    dark_red = (200, 0, 0)
    dark_green = (0, 200, 0)
    dark_blue = (0, 0, 200)

    x_init = 300
    y_init = 100
    x_span = 100
    y_span = 30
    thickness = 5
    pygame.init()
    display_width = 1000
    display_height = 600
    font = 'freesansbold.ttf'
    pygame.display.set_caption('Steel Locating')
    clock = pygame.time.Clock()
    pygame.key.set_repeat()
    button_goal = (display_width - 100, 10, 70, 40)

    def __init__(self, locating, width, height, num_block):
        self.width = width
        self.height = height
        self.num_block = num_block
        self.space = locating
        self.on_button = False
        self.total = 0
        self.display_width = 150 * width + 200
        self.gameDisplay = pygame.display.set_mode((self.display_width, self.display_height))

    def restart(self):
        self.space.reset()
        self.game_loop_from_space()

    def text_objects(self, text, font):
        text_surface = font.render(text, True, self.white)
        return text_surface, text_surface.get_rect()

    def block(self, x, y, text='', color=(0, 255, 0), x_init=100):
        pygame.draw.rect(self.gameDisplay, color, (int(x_init + self.x_span * x),
                                                   int(self.y_init + self.y_span * y),
                                                   int(self.x_span),
                                                   int(self.y_span)))
        large_text = pygame.font.Font(self.font, 20)
        text_surf, text_rect = self.text_objects(text, large_text)
        text_rect.center = (int(x_init + self.x_span * (x + 0.5)), int(self.y_init + self.y_span * (y + 0.5)))
        self.gameDisplay.blit(text_surf, text_rect)

    def board(self, step, reward=0):
        large_text = pygame.font.Font(self.font, 20)
        text_surf, text_rect = self.text_objects('step: ' + str(step) + '   reward: ' + format(reward, '.2f')
                                                 + '   total: ' + format(self.total, '.2f'), large_text)
        text_rect.center = (200, 20)
        self.gameDisplay.blit(text_surf, text_rect)

    def button(self, goal=0):
        color = self.dark_blue
        str_goal = 'In'
        if self.on_button:
            color = self.blue
        if goal == 0:
            str_goal = 'Out'
            color = self.dark_red
            if self.on_button:
                color = self.red
        pygame.draw.rect(self.gameDisplay, color, self.button_goal)
        large_text = pygame.font.Font(self.font, 20)
        text_surf, text_rect = self.text_objects(str_goal, large_text)
        text_rect.center = (int(self.button_goal[0] + 0.5 * self.button_goal[2]),
                            int(self.button_goal[1] + 0.5 * self.button_goal[3]))
        self.gameDisplay.blit(text_surf, text_rect)

    def game_loop_from_space(self):
        action = -1
        game_exit = False
        done = False
        reward = 0
        self.total = 0
        while not game_exit:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        action = 0
                    elif event.key == pygame.K_2:
                        action = 1
                    elif event.key == pygame.K_3:
                        action = 2
                    elif event.key == pygame.K_4:
                        action = 3
                    elif event.key == pygame.K_5:
                        action = 4
                    elif event.key == pygame.K_6:
                        action = 5
                    elif event.key == pygame.K_7:
                        action = 6
                    elif event.key == pygame.K_8:
                        action = 7
                    elif event.key == pygame.K_ESCAPE:
                        game_exit = True
                        break
                if action != -1:
                    _, reward, done = self.space.step(action)
                    self.total += reward
                if done:
                    self.restart()
                # click = pygame.mouse.get_pressed()
                # mouse = pygame.mouse.get_pos()
                self.on_button = False
                action = -1
            self.gameDisplay.fill(self.black)
            self.draw_space(self.space)
            self.board(self.space.stage, reward)
            self.draw_grid(self.width, 1, self.x_init, self.y_init, self.x_span, self.y_span * self.height)
            self.draw_grid(1, 1, 100, 100, self.x_span, self.y_span * 10)
            self.message_display('Inbound plates', 150, 80)
            self.message_display('Stockyard', 500, 80)
            pygame.display.flip()
            self.clock.tick(10)

    def draw_grid(self, width, height, x_init, y_init, x_span, y_span):
        pygame.draw.line(self.gameDisplay, self.blue, (x_init, y_init),
                         (x_init, y_init + y_span * height), self.thickness)
        pygame.draw.line(self.gameDisplay, self.blue, (x_init, y_init),
                         (x_init + x_span * width, y_init), self.thickness)
        for i in range(width):
            pygame.draw.line(self.gameDisplay, self.blue, (x_init + x_span * (i + 1), y_init),
                             (x_init + x_span * (i + 1), y_init + y_span * height), self.thickness)
        for i in range(height):
            pygame.draw.line(self.gameDisplay, self.blue, (x_init, y_init + y_span * (i + 1)),
                             (x_init + x_span * width, y_init + y_span * (i + 1)), self.thickness)

    def draw_space(self, space):
        for i, pile in enumerate(space.plates):
            for j, plate in enumerate(pile):
                rgb = 150 * (1 / max(1, plate.outbound - space.current_date))
                self.block(i, self.space.max_stack - j - 1, plate.id, (rgb, rgb, rgb), x_init=self.x_init)
        for i, plate in enumerate(space.inbound_plates[:10]):
            rgb = 150 * (1 / max(1, plate.outbound - plate.inbound))
            self.block(0, self.space.max_stack - i - 1, plate.id, (rgb, rgb, rgb), x_init=100)

    def message_display(self, text, x, y):
        large_text = pygame.font.Font(self.font, 20)
        text_surf, text_rect = self.text_objects(text, large_text)
        text_rect.center = (x, y)
        self.gameDisplay.blit(text_surf, text_rect)


# 환경 가시화 및 테스트시에 사용하는 코드
if __name__ == '__main__':
    #inbounds = [plate.Plate('P' + str(i), outbound=-1) for i in range(30)]  # 테스트용 임의 강재 데이터
    #inbounds = plate.import_plates_schedule('data/plate_example1.csv')

    inbounds = plate.import_plates_schedule('data/SampleData.csv')
    s = Locating(max_stack=10, num_pile=8, inbound_plates=inbounds, display_env=True)  # 환경 테스트
    print(s.plates)