from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, LSTM, GRU
import numpy as np
from tkinter import Tk, Canvas, Label, Button
from tkinter.filedialog import *
import math
import time
import threading
import random
#from tkFileDialog import askopenfilename


class obstacle():
    '''class for obstacle objects
    '''

    def __init__(self, x, y, can):
        self.x = x
        self.y = y
        self.color = 'orange'
        self.size = 50
        self.obstacle_object = can.create_rectangle(
            self.x, self.y, self.x + self.size, self.y + self.size,
            fill=self.color, outline='black')


class target():
    '''class for target objects
    '''

    def __init__(self, can):
        # inside 200 600 outside 600 200 center
        self.x = 520
        self.y = 80

        self.size = 200
        self.color = 'green'
        self.win_range = (self.size / 2)
        self.x_center = self.x + (self.size / 2)
        self.y_center = self.y + (self.size / 2)
        self.target_object = can.create_oval(
            self.x, self.y, self.x + self.size,
                            self.y + self.size,
            fill=self.color, outline='black')


class world():
    '''large class describes agent, world, mechanics
    '''

    def __init__(self):
        '''world properties
        '''
        self.size = 800
        self.center = self.size / 2
        # for reset
        self.x_start = 200
        self.y_start = 700

        self.x = self.x_start
        self.y = self.y_start

        # game counters
        self.wins = 0
        self.games = 0
        self.reward_sum = 0
        # robot properties
        self.body_size = 20
        self.eye_size = 300

        self.step = 15  # cm
        self.step_angle = 20  # degrees
        # main angle
        self.angle_start = -90
        self.angle = self.angle_start

        self.eyes_positions = [-40,
                               -20,
                               0,
                               20,
                               40]
        # creating world
        self.root = Tk()
        self.can = Canvas(self.root, width=self.size, height=self.size, bg='snow')
        self.can.pack()
        self.label = Label(self.root)
        self.label.pack(side='bottom')

        self.button_start = Button(self.root, text='start', command=self.main)
        self.button_start.pack()
        self.button_save = Button(self.root, text='save nn', command=self.save_model)
        self.button_save.pack()
        self.button_stop = Button(self.root, text='stop', command=self.stop_game)
        self.button_stop.pack()
        self.button_load = Button(self.root, text='load', command=self.load_nn)
        self.button_load.pack()

        self.game_thread = None

        self.thread_bool = False
        self.load_bool = False
        self.load = None
        # target creating
        self.target = target(self.can)
        # obstacles creating
        self.obstacles = []
        for i in range(3):
            obs = obstacle(x=400, y=650 - (i * (50)), can=self.can)
            self.obstacles.append(obs)
        for i in range(3):
            obs = obstacle(x=250 - (i * (50)), y=300, can=self.can)
            self.obstacles.append(obs)
        i = 0
        while (i != 800):
            self.obstacles.append(obstacle(x=i, y=750, can=self.can))
            i += 50
        i = 0
        while (i != 800):
            self.obstacles.append(obstacle(x=750, y=i, can=self.can))
            i += 50
        i = 0
        while (i != 800):
            self.obstacles.append(obstacle(x=0, y=i, can=self.can))
            i += 50
        i = 0
        while (i != 800):
            self.obstacles.append(obstacle(x=i, y=0, can=self.can))
            i += 50
        # robot creating
        # BRAIN (AI)
        self.nn = None
        self.memory = []
        self.memory_limit = 50000
        self.move_forward_flag = False
        self.rotate_flag = False
        # constants for epsilon-greedy algorithms
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99995
        # body
        self.body = self.can.create_oval(self.body_coords(self.x, self.y), fill='yellow', outline='black')
        # sensors(eyes)
        self.eyes = []
        for pos in self.eyes_positions:
            self.eyes.append(
                [self.can.create_line(self.eye_coords(self.x, self.y, self.angle - pos, self.eye_size), width=1),
                 pos])
        # sensor`s range
        #
        # |\
        # 1| \ 3
        # |  \
        # -----
        #   2
        self.range = self.range_triangle(self.x, self.y, angle=self.angle, eye_size=self.eye_size)

    # AI methods
    def remember(self, state, action, reward, state_, done):
        '''appending datasets in memory
        '''
        self.memory.append([state, action, reward, state_, done])

    def prepare_data(self, arr, prepare_type=0):
        '''normalize data for NN`s input
        arr - data from 5 eye-sensors,
        (x, y) coords and angle [8 params]
        ------params
        prepare_type: 1 - returns data without any changes
                      0 - returns data normalize in range[0,1]
        '''

        data = []

        if prepare_type == 0:
            for i in arr[: 5]:
                data.append(np.round(float(i) / self.eye_size, 2))
            for i in arr[5: 7]:
                data.append(np.round(float(i) / self.size, 2))
            i = arr[-1]
            # dividing by 360 - full rotate angle
            data.append(round(float(i) / 360, 2))
            data = np.asarray(data)
            # data = np.reshape(data, (1, 8))

        if prepare_type == 1:
            if np.shape(arr) == (1, 8):
                for j in arr:

                    for i in j[: 5]:
                        data.append(np.round(float(i), 2))
                    for i in j[5: 7]:
                        data.append(np.round(float(i), 2))
            i = arr[0][-1]

            data.append(round(float(i), 2))
            data = np.asarray(data)

        return np.reshape(data, (1, 1, 8))

    def generate_NN(self):
        '''creating RNN
        for DQN
        '''
        if self.load_bool == False:
            model = Sequential()

            model.add(GRU(100, input_shape=(1, 8)))
            model.add(Dense(3))
            model.add(Activation('softmax'))

            model.compile(optimizer='adam', loss='mse')
        if self.load_bool == True:
            model = self.load
        return model

    def retrain(self, model, memory, batchsize=4, gamma=0.9):
        '''method for retrain NN by DQN algorithm
        creates minibatch from memory`s dataset
        uses minibatch for retrain
        '''
        batch = batchsize
        l = len(memory)
        if l < batchsize:
            batch = l
        minibatch = random.sample(memory, batch)
        for s, a, r, s_, d in minibatch:
            target = r
            state = self.prepare_data(s[0], 1)
            state_ = self.prepare_data(s_[0], 1)
            if not d:
                target = r + gamma * np.amax(model.predict(state_))

            target_f = model.predict(state)
            target_f[0][a] = target
            model.fit(state, target_f, nb_epoch=1, verbose=0)

    def act(self, state, model):
        '''returns action based on current state
        '''
        if np.random.rand() <= self.epsilon:
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay
            return random.randint(0, 2)
        prediction = model.predict(state)

        return np.argmax(prediction)

    def harvest_data(self):
        '''collects data
        [e1 e2 e3 e4 e5] x y angle r done
        '''
        data, reward, done = self.math_factory()
        x, y = self.x, self.y

        angle = self.angle

        return data, x, y, angle, reward, done

    def move_AI(self, action):
        '''translates AI`s signal into move
        '''
        if action == 0:
            self.rotate_right()
            self.rotate_flag = True
        if action == 1:
            self.rotate_left()
            self.rotate_flag = True
        if action == 2:
            self.go_forward()
            self.move_forward_flag = True

        if action == 3:
            self.go_backward()

    def check_memory_size(self):
        '''memory must have stable size
        deletes first elements if its to large
        '''
        if len(self.memory) > self.memory_limit:
            self.memory.pop(0)

    def init_AI(self):
        self.nn = self.generate_NN()

    # ------------------------------------------------------------RUNURNURNURNRU
    def run_AI(self, AI):
        '''method for main computation
        '''

        data, x, y, angle, reward, done = self.harvest_data()
        state_arr = data

        state_arr.append(x)
        state_arr.append(y)
        state_arr.append(angle)

        state = self.prepare_data(arr=state_arr, prepare_type=0)

        action = self.act(state=state, model=AI)
        self.move_AI(action=action)

        data_, x_, y_, angle_, reward_, done_ = self.harvest_data()

        state_arr_ = data_

        state_arr_.append(x_)
        state_arr_.append(y_)
        state_arr_.append(angle_)

        state_ = self.prepare_data(arr=state_arr_)

        self.remember(state=state, action=action, reward=reward_, state_=state_, done=done_)
        # checking for a memory size
        self.check_memory_size()
        self.retrain(model=AI, memory=self.memory, batchsize=2)

    # info write methods

    def write_info(self, **data):
        '''writing info on gui in format:
        name = 123 -> 'name : 123'
        '''
        sum_info = []
        for info in data:
            inf = str(info) + ': ' + str(data[info])
            sum_info.append(inf)
        sum_data = ''
        for text_data in sum_info:
            sum_data += text_data + '\n'
        self.label.config(text=sum_data)

        # math methods

    def pos_angle(self):
        '''Checking if main angle less than 0 recount to positive
        '''
        if self.angle < 0:
            self.angle += 360

    def body_coords(self, x, y):
        '''getting agent`s coords
        '''
        b_x1 = x - self.body_size
        b_y1 = y - self.body_size
        b_x2 = x + self.body_size
        b_y2 = y + self.body_size
        return b_x1, b_y1, b_x2, b_y2

    def body_center(self):
        '''returns body`s center coords
        '''
        return self.x + (self.body_size / 2), self.y + (self.body_size / 2)

    def range_triangle(self, x, y, angle, eye_size):
        '''calculates range of help triangle
        '''
        size = eye_size * math.sqrt(2)
        r1 = self.can.create_line(self.eye_coords(x=x, y=y, angle=angle + 45, size=size), width=2)
        r2 = self.can.create_line(self.eye_coords(x=x, y=y, angle=angle - 45, size=size), width=2)
        return [[r1, 45], [r2, - 45]]

    def eye_coords(self, x, y, angle, size):
        '''returns coords of 2 points
        from start point (x, y), angle and len
        '''
        e_x1 = x
        e_y1 = y
        angle_rad = math.radians(angle)
        e_x2 = x + ((size * math.cos(angle_rad)))
        e_y2 = y + ((size * math.sin(angle_rad)))
        return e_x1, e_y1, e_x2, e_y2

    def check_angle(self):
        '''method for normolize angle in
         [-360, 360] limits
        '''
        if (self.angle >= 360):
            self.angle -= 360
        if (self.angle <= -360):
            self.angle += 360

    def get_triangle_square(self, x1, y1, x2, y2, x3, y3):
        '''calculates square of helps triangle
        '''
        square = 0.5 * ((x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3))
        return square

    def check_scan(self):
        '''checking arr with objects
        from help triangle
        '''
        obs_buff = []
        x11, y11 = self.x, self.y
        x21, y21, x22, y22 = self.can.coords(self.range[0][0])
        x31, y31, x32, y32 = self.can.coords(self.range[1][0])
        for obs in self.obstacles:
            point1 = (obs.x, obs.y)
            point2 = (obs.x + obs.size, obs.y + obs.size)
            point3 = (obs.x, obs.y + obs.size)
            point4 = (obs.x + obs.size, obs.y)
            points = (point1, point2, point3, point4)
            for p in points:
                b1 = (x11 - p[0]) * (y22 - y11) - (x22 - x11) * (y11 - p[1])
                b2 = (x22 - p[0]) * (y32 - y22) - (x32 - x22) * (y22 - p[1])
                b3 = (x32 - p[0]) * (y11 - y32) - (x11 - x32) * (y32 - p[1])
                if ((b1 > 0 and b2 > 0 and b3 > 0) or (b1 < 0 and b2 < 0 and b3 < 0)):
                    obs_buff.append(obs)
                    break
        return obs_buff

    def line_equation(self, x1, y1, x2, y2):
        '''returns line equation`s koeffs
        '''
        a = y1 - y2
        b = x2 - x1
        c = (x1 * y2) - (x2 * y1)
        return a, b, c

    def rect_equations(self, x1, y1, x2, y2):
        '''returns 4 rectangle`s line equations
        '''
        # 1
        x11 = x1
        y11 = y1
        x12 = x2
        y12 = y1
        # 2
        x21 = x2
        y21 = y1
        x22 = x2
        y22 = y2
        # 3
        x31 = x2
        y31 = y2
        x32 = x1
        y32 = y2
        # 4
        x41 = x1
        y41 = y2
        x42 = x1
        y42 = y1
        # 4 equations
        rect_eq1 = self.line_equation(x11, y11, x12, y12)
        rect_eq2 = self.line_equation(x21, y21, x22, y22)
        rect_eq3 = self.line_equation(x31, y31, x32, y32)
        rect_eq4 = self.line_equation(x41, y41, x42, y42)
        return [[[x11, y11, x12, y12], rect_eq1], [[x21, y21, x22, y22], rect_eq2],
                [[x31, y31, x32, y32], rect_eq3], [[x41, y41, x42, y42], rect_eq4]]

    def solve_eq(self, eq1, eq2):
        '''method for equation solving
        returns point - result
        '''
        #    |A1 B1|
        #    |     | = 0 if this -> parallel!
        #    |A2 B2|
        parallel = (eq1[0] * eq2[1]) - (eq2[0] * eq1[1])
        if (parallel != 0):
            det = (eq1[0] * eq2[1]) - (eq2[0] * eq1[1])
            det_x = (-eq1[2] * eq2[1]) - (-eq2[2] * eq1[1])
            det_y = (eq1[0] * (-eq2[2])) - (eq2[0] * (-eq1[2]))
            x = det_x / det
            y = det_y / det
            return x, y

    def unique(self, arr):
        '''filters same coords in arr
        '''
        buff = []
        for i in arr:
            if i not in buff:
                buff.append(i)
        return buff

    def solve_eq_by_coords(self, a, b, c, x, y):
        '''equation solver by coords
        '''
        answer = False
        if (round(((a * x) + (b * y) + c), 8) == 0):
            answer = True
        return answer

    def find_module(self, x1, y1, x2, y2):
        '''calculates module between
        2 points by their coords
        '''
        module = math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))
        return module

    def find_intersection(self, line, square):
        '''finds inersection points
        between line and square
        '''
        line_coords = self.can.coords(line)
        square_coords = self.can.coords(square)
        line_k = self.line_equation(*line_coords)
        rect_k = self.rect_equations(*square_coords)
        intersections = []
        for coords, equation_k in rect_k:
            x1, y1 = coords[0:2]
            x2, y2 = coords[2:4]
            try:
                x, y = self.solve_eq(eq1=line_k, eq2=equation_k)
                m = self.find_module(x1=x1, y1=y1, x2=x2, y2=y2)
                m1 = self.find_module(x1=x1, y1=y1, x2=x, y2=y)
                m2 = self.find_module(x1=x, y1=y, x2=x2, y2=y2)
                if ((m1 + m2) == m):
                    intersections.append([x, y])
            except:
                pass
        intersections = self.unique(intersections)
        return intersections

    def processing_minimum_modules_vectors(self, intersection_coords_arr):
        '''calculates minimum module
        shorter distance from zero eye point
        to obstacles
        '''
        buff = []
        for eye, ang in self.eyes:
            eye_A, eye_B, eye_C = self.line_equation(*self.can.coords(eye))
            coords_arr = []
            for i in intersection_coords_arr:
                for x, y in i:
                    ans = self.solve_eq_by_coords(a=eye_A, b=eye_B, c=eye_C, x=x, y=y)
                    if (ans == True):
                        coords_arr.append([x, y])
            buff.append(coords_arr)
        # counting modules
        eyes_ranges = []
        for part in buff:
            sizes = []
            for x, y in part:
                # showing a true dist that gonna be counted
                module = math.sqrt(((self.x - x) ** 2) + ((self.y - y) ** 2))
                sizes.append(module)
            try:
                eyes_ranges.append(min(sizes))
            except:
                eyes_ranges.append(None)
        try:
            for range in eyes_ranges:
                if (range > self.eye_size):
                    index = eyes_ranges.index(range)
                    eyes_ranges.remove(range)
                    eyes_ranges.insert(index, None)
        except:
            pass
        return eyes_ranges

    # Game`s rules
    def check_crash(self, sensors, flag):
        '''if limit is flag so range less than flag is mean crash of robot
        '''
        crash = False
        for value in sensors:
            if ((type(value) is int) or (type(value) is float)):
                if (round(value, 3) <= flag):
                    crash = True
        return crash

    def check_finish(self, flag):
        '''checking the finish
        ----
        arguments:
        flag - if mudule less than flag - finish true
        (positive int)
        '''
        finish = False
        x, y = self.body_center()
        module = self.find_module(x1=x, y1=y, x2=self.target.x_center, y2=self.target.y_center)
        if (module <= flag):
            finish = True
            self.wins += 1
        return finish

    def reset(self):
        '''resets agent to default state
        '''
        self.games += 1
        self.x = self.x_start
        self.y = self.y_start

        self.angle = self.angle_start

        self.can.coords(self.body, *self.body_coords(self.x, self.y))
        i = 0
        for eye, angle in self.eyes:
            self.can.coords(eye, *self.eye_coords(self.x, self.y, self.angle + self.eyes_positions[i], self.eye_size))
            i += 1

        i = 0
        ranges = [-45, 45]
        for r, ang in self.range:
            # self.can.coords(r,*self.eye_coords(self.x, self.y,ang,self.range*math.sqrt(2)))
            self.can.coords(r,
                            self.eye_coords(self.x, self.y, self.angle + ranges[i], size=self.eye_size * math.sqrt(2)))
            i += 1

    def get_reward(self, flag):
        '''method for taking reward
        '''
        if flag == 'obs':
            return -20
        if flag == 'win':
            return 100
        if flag == 'move_forward':
            return -0.11
        if flag == 'move':
            return 0
        if flag == 'rotate':
            return -0.21

    # main math module

    def math_factory(self):
        '''method for main calculations
        returns data with eyes info and reward
        '''
        reward = 0
        self.pos_angle()
        # this reward will taking for moving forward
        if self.move_forward_flag == True:
            reward = self.get_reward(flag='move_forward')
        if self.rotate_flag == True:
            reward = self.get_reward(flag='rotate')
        done = False

        self.move_forward_flag = False
        self.rotate_flag = False

        self.check_angle()
        buff = self.check_scan()  # buff is here
        intersections = []
        for b in buff:
            for eye, an in self.eyes:
                intersections.append(self.find_intersection(line=eye, square=b.obstacle_object))
        result = self.processing_minimum_modules_vectors(intersection_coords_arr=intersections)
        data = []
        for i in result:
            if i is None:
                data.append(self.eye_size)
            else:
                data.append(i)
        # 13= 10(body radius) +  5(minimum range to body)
        # checking crash

        # flag for range to obstacle from sensor to crash
        if (self.check_crash(sensors=result, flag=18) == True):
            reward = self.get_reward('obs')
            self.reset()
        # checking finish
        if (self.check_finish(flag=(self.target.win_range + (self.body_size / 2)))) == True:
            reward = self.get_reward('win')

            done = True
            self.reset()
        # WRITING INF IN GUI
        self.write_info(eyes=data,
                        coord_x=self.x,
                        coord_y=self.y,
                        angle=self.angle,
                        games=self.games,
                        wins=self.wins,
                        total_reward=self.reward_sum)
        self.reward_sum += reward
        return data, reward, done

    # agent`s actions
    def rotate_left(self):

        self.angle -= self.step_angle
        for eye, ang in self.eyes:
            ang -= self.angle
            self.can.coords(eye, self.eye_coords(self.x, self.y, -ang, size=self.eye_size))
        for r, ang in self.range:
            ang -= self.angle
            self.can.coords(r, self.eye_coords(self.x, self.y, angle=- ang, size=self.eye_size * math.sqrt(2)))

    def rotate_right(self):
        self.angle += self.step_angle
        for eye, ang in self.eyes:
            ang += self.angle
            self.can.coords(eye, self.eye_coords(self.x, self.y, ang, size=self.eye_size))
        for r, ang in self.range:
            ang += self.angle
            self.can.coords(r, self.eye_coords(self.x, self.y, angle=ang, size=self.eye_size * math.sqrt(2)))

    def go_forward(self):
        self.math_factory()
        dx = (self.step * math.cos(math.radians(self.angle)))
        dy = (self.step * math.sin(math.radians(self.angle)))
        self.x += dx
        self.y += dy
        self.can.move(self.body, dx, dy)
        for eye, angle in self.eyes:
            self.can.move(eye, dx, dy)
        for r, ang in self.range:
            self.can.move(r, dx, dy)

    def go_backward(self):
        dx = - (self.step * math.cos(math.radians(self.angle)))
        dy = - (self.step * math.sin(math.radians(self.angle)))
        self.x += dx
        self.y += dy
        self.can.move(self.body, dx, dy)
        for eye, angle in self.eyes:
            self.can.move(eye, dx, dy)
        for r, ang in self.range:
            self.can.move(r, dx, dy)

    # moving algorithms for testing
    def rand_move(self):
        '''generates random action
        '''
        r = random.randint(0, 3)
        self.check_angle()
        if r == 0:
            self.go_forward()
        if r == 1:
            self.go_backward()
        if r == 2:
            self.rotate_left()
        if r == 3:
            self.rotate_right()

    def simple_move(self):
        '''left side movement
        '''
        i = 0
        t = 0.2

        def sleep(time_):
            time.sleep(time_)

        while (i != 10):
            sleep(t)
            self.go_forward()
            i += 1
        i = 0
        while (i != 30):
            sleep(t)
            self.rotate_left()
            i += 1
        self.check_angle()

    # ____________________________________________________________________
    def save_model(self, name='robot_RNN.h5'):
        self.nn.save(name)

    # main thread
    def load_nn(self):
        self.load = askopenfilename()
        self.load_bool = True
        # self.nn = load_model(op)

    def stop_game(self):
        self.thread_bool = True

    def run_graphics(self):
        self.root.mainloop()

    def main(self):
        self.game_thread = threading.Thread(target=self.run)
        self.game_thread.start()

    def run(self):
        self.nn = self.generate_NN()
        while True:
            if self.thread_bool == False:
                self.run_AI(AI=self.nn)

            if self.thread_bool == True:
                self.thread_bool = False
                break


w = world()
w.run_graphics()




