from GUI_tools import *
import numpy as np

# GUI and main loop

class RoomBuilder(Frame):
    # main gui

    def __init__(self, parent, *args, **kwargs):
        Frame.__init__(self, parent, *args, **kwargs)
        # initialise values
        self.root = parent
        self.room = None
        self.weather = np.array([])
        self.windows = []
        self.dict = {}

        # initialise advanced options
        self.wind_f = 0.1
        self.min_vent = 5
        self.occ_gain = 70
        self.vert_strat = 0
        self.occ_height = 1.6
        self.relax_assump = 0

        self.init_gui()

    def buildRoom(self):
        # builds room from GUI inputs

        # dimensions and materials
        n = self.room_name.get()
        a = float(self.room_area.get())
        h = float(self.room_height.get())
        # ceil = materials.loadmat(self.ceil_mat.get())
        # walls = materials.loadmat(self.wall_mat.get())
        # floor = materials.loadmat(self.floor_mat.get())

        self.room = build.Room(n, a, h, ceil, walls, floor) # build room object
        for win in self.windows:
            self.room.addWindow(win)    # add each window from GUI windows list to room
        self.room.windFactor = self.wind_f

        # add heat profiles
        occ = []
        add = []
        for i in range(24):
            tempocc = float(self.occ[i].get())  # occupancy heat gains for this hour
            occ.append(tempocc)

            # additional/lighting gains only applied when room is occupied
            if tempocc > 0:    # check if room is occupied (not occupied if heat gains = 0)
                add.append(float(self.additional.get()) + float(self.light.get()) * a)
            else:
                add.append(0.0)
        self.room.addOccupancy(occ, gain=self.occ_gain)
        self.room.addGain(add, 'additional')

        # add units
        for unit in range(int(self.no_unit.get())):
            # TODO: generalise this to all units
            if self.unit_type.get() == 'NVHR':
                self.room.addUnit(build.NVHR())
            else:
                self.room.addUnit(build.S1500())

        self.room.addControl(build.Control(int(self.nc_start.get()), int(self.nc_stop.get()), self.min_vent))
        self.room.temp['vert_strat'] = self.vert_strat
        self.room.occHeight = self.occ_height
        self.testlabel['text'] = self.room.name

    def unitSelect(self, event=None):
        # GUI config for selecting different units
        if self.unit_type.get() == 'NVHR':
            state = DISABLED
        else:
            state = NORMAL

        self.rt.config(state=state)
        self.rt_label.config(state=state)

    def winSelect(self, event=None):
        # GUI changes when different window types are selected
        # select custom for any opening type

        if self.win_type.get() == 'Custom':
            label = 'Specified A*:'
            state = DISABLED
            self.custom_flag = 1
        else:
            label = 'Opening Width:'
            state = NORMAL
            self.custom_flag = 0

        self.win_op_width_label['text'] = label
        self.win_orient_label.config(state=state)
        self.win_orient.config(state=state)
        self.win_gval_label.config(state=state)
        self.win_gval.config(state=state)
        self.win_area.config(state=state)
        self.win_area_label.config(state=state)
        self.restrict_check.config(state=state)
        self.restrict_amount.config(state=state)
        self.restrict_mm.config(state=state)

    def addWindow(self):
        # adding a window to the GUI window list
        a = float(self.win_area.get()) if not self.custom_flag else 0  # glazing area - custom opening has no glazing
        h = float(self.win_height.get())
        gval = float(self.win_gval.get())
        o = self.win_orient.get()
        op_w_A = float(self.win_op_W_or_Astar.get())
        op_h = float(self.win_op_height.get())
        res = float(self.restrict_amount.get())/1000 if self.restrict.get() else 0
        lb, ub = winAstar(op_w_A, op_h, type=self.win_type.get(), restrict=res)
        Astar = ub if self.relax_assump else lb     # use upper bound when assumptions are relaxed
        # TODO: relax assumptions will not change windows that were previously loaded up

        self.windows.append(build.Window(a, o, height=h, g_value=gval, specified_astar=Astar, opening_height=op_h))

        # window info applied to GUI. Note that win_list is just info while self.windows actually contains the data
        if self.custom_flag:
            self.win_list.insert(END, '{}, {}, A* = {}m2'.format(self.win_type.get(), o, round(Astar, 3)))
        else:
            self.win_list.insert(END, '{}, {}m2, A* = {}m2'.format(o, a, round(Astar, 3)))

    def deleteWindow(self):
        # deletes window from list
        del self.windows[self.win_list.curselection()[0]]
        self.win_list.delete(self.win_list.curselection()[0])

    def loadWeather(self):
        # loads weather file csv
        self.weather_label['text'] = 'Loading file...'
        self.update_idletasks()
        weather_str = self.weather_file.get() + ' ' + self.weather_file_type.get()  # [Location] + [DSY or TRY]
        self.weather = np.genfromtxt('weather/' + weather_str + '.csv', delimiter=',')[2900:6572, 1:]  # summer only
        self.weather_label['text'] = weather_str

    def advancedOptions(self):
        # talks to advanced options window
        default_vals = [str(self.wind_f), str(self.min_vent), str(self.occ_gain), self.vert_strat, str(self.occ_height), self.relax_assump]
        adv_opts = AdvancedOptions(self, defvals=default_vals, title='Advanced Options')    # create new window
        if adv_opts.vals:
            self.wind_f = adv_opts.vals[0]
            self.min_vent = adv_opts.vals[1]
            self.occ_gain = adv_opts.vals[2]
            self.vert_strat = adv_opts.vals[3]
            self.occ_height = adv_opts.vals[4]
            self.relax_assump = adv_opts.vals[5]

    def solveRoom(self, event=None):
        # solves room
        self.buildRoom()    # Build room
        if not self.weather.size:
            self.loadWeather()  # Load weather if not already loaded
        self.solve_status['text'] = 'Solving...'
        self.update_idletasks()     # updates GUI
        res1, res2, res3, res4, res5 = res.psbp(solve.solve(self.room, self.weather))   # solve + PSPB results

        # TODO: add other results and generalise below conditional formatting
        self.solve_status['text'] = 'Solved'
        self.hours_above_Tacc['text'] = str(res1)
        self.hours_above_Tacc.configure(foreground='green' if res1 <= 40 else 'red')
        self.weight_ex['text'] = str(res2)
        self.weight_ex.configure(foreground='green' if res2 <= 6 else 'red')
        self.hours_above_Tul['text'] = str(res3)
        self.hours_above_Tul.configure(foreground='green' if res3 == 0 else 'red')
        self.max_dT['text'] = str(res4)
        self.max_dT.configure(foreground='green' if res4 < 5 else 'red')
        self.result['text'] = res5
        self.result.configure(foreground='green' if res5 == 'PASS' else 'red')  # PSBP only


    def init_gui(self):
        # all the gui objects - ttk is incredibly wordy so there is a lot of code here but it's all quite simple
        # TODO: create abstract functions to prevent the same patterns of code being repeated for each GUI object
        self.root.title('Room Solver')
        # Everything is set to an imaginary grid within its frame, this is the main GUI frame
        self.grid(column=0, row=0)

        # ------------Room--------------------

        # Each section of the GUI has its own frame, with its own grid
        self.room_frame = Labelframe(self, text='Room')
        self.room_frame.grid(column=0, row=0, columnspan=2)     # placing the Roomn frame on the main GUI grid

        Label(self.room_frame, text='Room name:').grid(row=0, column=0)     # placing the label on the Room frame grid
        self.room_name = Entry(self.room_frame, width=10)
        self.room_name.grid(row=0, column=1)
        self.room_name.insert(0, 'Room 1')

        Label(self.room_frame, text='Area:').grid(row=1, column=0)
        self.room_area = Entry(self.room_frame, width=10)
        self.room_area.grid(row=1, column=1)
        self.room_area.insert(0, '55')

        Label(self.room_frame, text='Height:').grid(row=2, column=0)
        self.room_height = Entry(self.room_frame, width=10)
        self.room_height.grid(row=2, column=1)
        self.room_height.insert(0, '3')

        Label(self.room_frame, text="Ceiling Material").grid(row=0, column=2)
        self.ceil_mat = Combobox(self.room_frame, state='readonly')
        self.ceil_mat.grid(row=0, column=3)


        Label(self.room_frame, text="Wall Material").grid(row=1, column=2)
        self.wall_mat = Combobox(self.room_frame, state='readonly')
        self.wall_mat.grid(row=1, column=3)


        Label(self.room_frame, text="Floor Material").grid(row=2, column=2)
        self.floor_mat = Combobox(self.room_frame, state='readonly')
        self.floor_mat.grid(row=2, column=3)


        # ---------------------------------------
        # -------------Window/openings--------------------

        self.custom_flag = 0

        self.win_frame = Labelframe(self, text='Windows/Openings')
        self.win_frame.grid(column=0, row=1, columnspan=2, rowspan=2)

        Label(self.win_frame, text='Type:').grid(row=0, column=0)
        self.win_type = Combobox(self.win_frame, values=('Top hung', 'Bottom hung', 'Side hung', 'Sliding', 'Custom'),
                                   state='readonly', width=12)
        self.win_type.bind("<<ComboboxSelected>>", self.winSelect)
        self.win_type.grid(row=0, column=1, pady=5)
        self.win_type.set('Top hung')

        self.restrict = IntVar()
        self.restrict_check = Checkbutton(self.win_frame, text="Restricted", variable=self.restrict)
        self.restrict_check.grid(row=0,column=3)
        self.restrict.set(0)

        self.restrict_amount = Entry(self.win_frame, width=3)
        self.restrict_amount.grid(row=0, column=4, sticky='e')
        self.restrict_amount.insert(0,'100')
        self.restrict_mm = Label(self.win_frame, text='mm')
        self.restrict_mm.grid(row=0, column=5, sticky='w')

        self.win_op_width_label = Label(self.win_frame, text='Opening Width:')
        self.win_op_width_label.grid(row=1, column=0)
        self.win_op_W_or_Astar = Entry(self.win_frame, width=5)
        self.win_op_W_or_Astar.grid(row=1, column=1)
        self.win_op_W_or_Astar.insert(0, '2.5')

        Label(self.win_frame, text='Opening Height:').grid(row=2, column=0)
        self.win_op_height = Entry(self.win_frame, width=5)
        self.win_op_height.grid(row=2, column=1)
        self.win_op_height.insert(0, '2')

        Label(self.win_frame, text='Sill height:').grid(row=3, column=0)
        self.win_height = Entry(self.win_frame, width=5)
        self.win_height.grid(row=3, column=1)
        self.win_height.insert(0, '1')

        self.win_area_label = Label(self.win_frame, text='Glazing Area:')
        self.win_area_label.grid(row=4, column=0)
        self.win_area = Entry(self.win_frame, width=5)
        self.win_area.grid(row=4, column=1)
        self.win_area.insert(0, '5')

        self.win_gval_label = Label(self.win_frame, text='G Value:')
        self.win_gval_label.grid(row=5, column=0)
        self.win_gval = Entry(self.win_frame, width=5)
        self.win_gval.grid(row=5, column=1)
        self.win_gval.insert(0, '0.4')

        self.win_orient_label = Label(self.win_frame, text='Orientation:')
        self.win_orient_label.grid(row=6, column=0)
        self.win_orient = Combobox(self.win_frame, values=('E', 'SE', 'S', 'SW', 'W', 'NW', 'N', 'NE', 'H'),
                                   state='readonly', width=3)
        self.win_orient.grid(row=6, column=1)
        self.win_orient.set('E')

        Separator(self.win_frame, orient=VERTICAL).grid(column=2, row=1, rowspan=7, sticky='ns', padx=5, pady=2)

        Label(self.win_frame, text='Window List').grid(row=1, column=3, columnspan=4)
        self.win_list = Listbox(self.win_frame, width=25)
        self.win_list.grid(row=2, column=3, rowspan=5, columnspan=4)

        Button(self.win_frame, text='Add', command=self.addWindow).grid(row=7, column=0, columnspan=2, pady=5)
        Button(self.win_frame, text='Delete', command=self.deleteWindow).grid(row=7, column=3, columnspan=2, pady=5)

        # -------------------------------------------------------------
        # ----------------------Occupancy and gains-------------------

        self.occ_frame = Labelframe(self, text='Occupancy and gains')
        self.occ_frame.grid(column=0, row=3, columnspan=4)

        Label(self.occ_frame, text='Occupied days per week').grid(row=0, column=0, columnspan=2)
        Label(self.occ_frame, text='Lighting (w/m2)').grid(row=0, column=4, columnspan=3)
        Label(self.occ_frame, text='Additional (w)').grid(row=0, column=9, columnspan=3)

        self.occ_days = Entry(self.occ_frame, width=2)
        self.occ_days.grid(row=0, column=2)
        self.occ_days.insert(0, '5')
        self.light = Entry(self.occ_frame, width=3)
        self.light.grid(row=0, column=7)
        self.light.insert(0, '10')
        self.additional = Entry(self.occ_frame, width=5)
        self.additional.grid(row=0, column=12, columnspan=2)
        self.additional.insert(0, '300')

        Separator(self.occ_frame, orient=HORIZONTAL).grid(column=0, row=1, columnspan=25, sticky='ew', padx=2, pady=5)

        self.occ = []
        self.cus_add = []

        self.default_occ = [0, 0, 0, 0, 0, 0, 0, 0, 0, 32, 32, 32, 2, 32, 32, 32, 0, 0, 0, 0, 0, 0, 0, 0]

        Label(self.occ_frame, text='Time').grid(row=2, column=0, padx=5)
        Label(self.occ_frame, text='Occupancy Profile (ppl)').grid(row=3, column=0, padx=5)
        # Label(self.occ_frame, text='Custom add. gains (w)').grid(row=4, column=0, padx=5)  # advanced option

        for i in range(24):
            Label(self.occ_frame, text=str(100 * i)).grid(row=2, column=i + 1, sticky='w')
            self.occ.append(Entry(self.occ_frame, width=4))
            self.occ[i].grid(row=3, column=i + 1)
            self.occ[i].insert(0, self.default_occ[i])
            # self.cus_add.append(Entry(self.occ_frame, width=4))
            # self.cus_add[i].grid(row=4, column=i + 1)
            # self.cus_add[i].insert(0, 0)

        # --------------------------------------------------------------
        # ----------------------------Units------------------------------

        self.unit_frame = Labelframe(self, text='Units')
        self.unit_frame.grid(row=0, column=2)

        Label(self.unit_frame, text="Unit type").grid(row=0, column=0)
        self.unit_type = Combobox(self.unit_frame, values=('NVHR', 'S1500'), state='readonly', width=8)
        self.unit_type.set('NVHR')
        self.unit_type.bind("<<ComboboxSelected>>", self.unitSelect)
        self.unit_type.grid(row=0, column=1, pady=2)

        Label(self.unit_frame, text="Number of units").grid(row=1, column=0)
        self.no_unit = Entry(self.unit_frame, width=2)
        self.no_unit.grid(row=1, column=1)
        self.no_unit.insert(0, '2')

        self.rt_label = Label(self.unit_frame, text="Roof Terminal", state=DISABLED)
        self.rt_label.grid(row=2, column=0)
        self.rt = Combobox(self.unit_frame, values=('Mushroom','Penthouse','None'), state='readonly', width=12)
        self.rt.grid(row=2, column=1)
        self.rt.set('Mushroom')
        self.rt.config(state=DISABLED)


        # --------------------------------------------------------------
        # ----------------------------Control------------------------------

        self.contr_frame = Labelframe(self, text='Control')
        self.contr_frame.grid(row=1, column=2)

        Label(self.contr_frame, text="Night cool start").grid(row=0, column=0, padx=5)
        self.nc_start = Entry(self.contr_frame, width=4)
        self.nc_start.grid(row=0, column=1)
        self.nc_start.insert(0, '1800')

        Label(self.contr_frame, text="Night cool stop").grid(row=1, column=0, padx=5)
        self.nc_stop = Entry(self.contr_frame, width=4)
        self.nc_stop.grid(row=1, column=1)
        self.nc_stop.insert(0, '800')

        # --------------------------------------------------------------
        # ----------------------------Advanced Options------------------

        Button(self, text="Advanced Options", command=self.advancedOptions).grid(row=0, column=3)

        # --------------------------------------------------------------
        # ----------------------------Weather------------------------------

        self.weather_frame = Labelframe(self, text='Weather')
        self.weather_frame.grid(row=2, column=2)

        #Label(self.weather_frame, text="Weather file").grid(row=0, column=0, padx=5)
        self.weather_file = Combobox(self.weather_frame, values=('Belfast', 'Birmingham', 'Cardiff', 'Edinburgh',
                                    'Glasgow', 'Leeds', 'London', 'Manchester', 'Newcastle', 'Norwich', 'Nottingham',
                                    'Plymouth', 'Southampton', 'Swindon'), state='readonly', width=13)
        self.weather_file_type = Combobox(self.weather_frame, values=('DSY', 'TRY'), state='readonly', width=4)
        self.weather_file.grid(row=0, column=0, pady=2, padx=2)
        self.weather_file_type.grid(row=0, column=1, pady=2)
        self.weather_file.set('London')
        self.weather_file_type.set('DSY')

        Button(self.weather_frame, text='Load Weather', command=self.loadWeather).grid(row=1, column=0)
        self.weather_label = Label(self.weather_frame, text='[None]')
        self.weather_label.grid(row=1, column=1)

        # --------------------------------------------------------------
        # ----------------------------Solver------------------------------
        self.solve_frame = Labelframe(self, text='Solver')
        self.solve_frame.grid(row=4, column=0)

        Button(self.solve_frame, text='Build Room', command=self.buildRoom).grid(row=0, column=0)
        self.testlabel = Label(self.solve_frame, text='[no room]')
        self.testlabel.grid(row=0, column=1)

        Button(self.solve_frame, text='Solve Room', command=self.solveRoom).grid(row=1, column=0)
        self.solve_status = Label(self.solve_frame, text='[not solved]')
        self.solve_status.grid(row=1, column=1)

        # --------------------------------------------------------------
        # ----------------------------Results------------------------------
        self.res_frame = Labelframe(self, text='Results')
        self.res_frame.grid(row=4, column=1, columnspan=3)

        Label(self.res_frame, text='Hours above Tacc: ').grid(row=0, column=0)
        Label(self.res_frame, text='Weighted exceedance: ').grid(row=0, column=2)
        Label(self.res_frame, text='Hours above max T: ').grid(row=0, column=4)
        Label(self.res_frame, text='Max delta T: ').grid(row=0, column=6)
        Label(self.res_frame, text='PSBP result: ').grid(row=1, column=0)

        self.hours_above_Tacc = Label(self.res_frame, text='[not solved]')
        self.hours_above_Tacc.grid(row=0, column=1)
        self.weight_ex = Label(self.res_frame, text='[not solved]')
        self.weight_ex.grid(row=0, column=3)
        self.hours_above_Tul = Label(self.res_frame, text='[not solved]')
        self.hours_above_Tul.grid(row=0, column=5)
        self.max_dT = Label(self.res_frame, text='[not solved]')
        self.max_dT.grid(row=0, column=7)
        self.result = Label(self.res_frame, text='[not solved]')
        self.result.grid(row=1, column=1)

        # --------------------------------------------------------------
        # ----------------------------Bindings-----------------------------

        #self.bind('<Enter>', self.solveRoom)

        # --------------------------------------------------------------
        # -------------------------Testing------------------------------
        # This can be used to develop new features without breaking the rest of the program
        # Can also try out new GUI options here before having to rearrange the whole grid

        self.test_frame = Labelframe(self, text='Testing')
        self.test_frame.grid(row=1, column=3, rowspan=2)

        Label(self.test_frame, text='Test new features here').grid()

        Button(self.test_frame, text='Save Room', command=self.saveRoom).grid(row=5, column=0)
        self.save_status = Label(self.test_frame, text='[not saved]')
        self.save_status.grid(row=5, column=1)

        Button(self.test_frame, text='Solve Room', command=self.testSolveRoom).grid(row=6, column=0)
        self.solve_status2 = Label(self.test_frame, text='[not solved]')
        self.solve_status2.grid(row=6, column=1)

        Button(self.test_frame, text='Winter', command=self.winter).grid(row=7, column=0)
        self.solve_status3 = Label(self.test_frame, text='[not solved]')
        self.solve_status3.grid(row=7, column=1)

    def testSolveRoom(self):
        # for testing stuff - can be ignored
        # at the moment I use this for plotting various variables over the hottest day of the london weather file
        # TODO: implement custom plots into the program?

        fig, ax1 = plt.subplots()
        max_ts = 100
        test_ts = [2,8, 25, 50, max_ts]     # time step study
        test_ts = [8]
        maxt = []
        maxdayt = []
        meant = []
        for ts in test_ts:
            self.buildRoom()
            if not self.weather.size:
                self.loadWeather()
            self.solve_status2['text'] = 'Solving...'
            self.update_idletasks()
            res1, res2, res3, res4, res5 = res.psbp(tsolve.solve(self.room, self.weather, time_step=ts))
            self.solve_status2['text'] = 'Solved'
            self.hours_above_Tacc['text'] = str(res1)
            self.hours_above_Tacc.configure(foreground='green' if res1 <= 40 else 'red')
            self.weight_ex['text'] = str(res2)
            self.weight_ex.configure(foreground='green' if res2 <= 6 else 'red')
            self.hours_above_Tul['text'] = str(res3)
            self.hours_above_Tul.configure(foreground='green' if res3 == 0 else 'red')
            self.max_dT['text'] = str(res4)
            self.max_dT.configure(foreground='green' if res4 < 5 else 'red')
            self.result['text'] = res5
            self.result.configure(foreground='green' if res5 == 'PASS' else 'red')

            # x = range(len(self.room.temp['int_day'][100:200]))
            # ax1.plot(x, self.room.temp['int_day'][100:200])
            maxt.append(max(self.room.temp['int']))
            maxdayt.append(max(self.room.temp['int_day']))
            meant.append(np.mean(self.room.temp['int']))

            x = np.arange(0,24,1/ts)
            ax1.plot(x, self.room.temp['sample_day'], label=str(ts))

        ax1.plot(x, self.room.temp['sample_day_ext'], 'b-.', label='T ext')
        ax1.plot(x, self.room.temp['tmass'], 'm-.', label='T mass')
        ax2 = ax1.twinx()
        ax2.plot(x, self.room.temp['vent'], 'g--', label='vent')
        #ax2.plot(x, self.room.temp['leak'], 'c--', label='leak')
        #ax2.plot(x, self.room.temp['window_vent'], 'r--', label='window vent')

        x2 = range(24)
        # ax1.plot(x2, data[:24], 'b-.', label='T ext')
        # ax1.plot(x2, data[24:48], 'r+', label='From excel')

        # ax1.plot(test_ts, maxt, 'r')
        # ax1.plot(test_ts, maxdayt, 'g')
        # ax1.plot(test_ts, meant, 'b')
        # ax1.xlabel('Time steps per hour')
        ax1.set_ylabel('Temperature (degC)')
        ax1.set_xlabel('Hour')
        ax2.set_ylabel('Vent (m3/s)')

        plt.legend(loc='upper left')
        plt.show()

    def winter(self):
        # for calculating winter CO2 levels
        # Not sure if this should be implemented, as it probably won't agree with what we say the units can achieve
        self.buildRoom()
        file = self.weather_file.get() + ' ' + self.weather_file_type.get()
        weather, winter_flag = winter_vent(file)
        tsolve.solve(self.room, weather, winter=True, winter_flag=winter_flag)
        plt.plot(self.room.temp['vent_year'][:1000])
        plt.show()
        print(self.room.temp['failflag'])

    def saveRoom(self):
        # testing saving rooms - this is properly unfinished
        with open('rooms.txt', 'w') as outfile:
            json.dump(self.room, outfile)


class AdvancedOptions(Dialog):
    # for options that generally stay the same
    # can add new options here

    def __init__(self, parent, defvals, title=None):
        self.vals = None
        self.defvals = defvals
        super().__init__(parent, title=title)

    def body(self, master):

        Label(master, text="Wind factor:").grid(row=0)
        self.wind_f = Entry(master)
        self.wind_f.grid(row=0, column=1)
        self.wind_f.insert(0,self.defvals[0])
        Label(master, text="Minimum vent (l/s per person):").grid(row=1)
        self.min_vent = Entry(master)
        self.min_vent.grid(row=1, column=1)
        self.min_vent.insert(0,self.defvals[1])
        Label(master, text="Gain per Occupant (w):").grid(row=2)
        self.occ_gain = Entry(master)
        self.occ_gain.grid(row=2, column=1)
        self.occ_gain.insert(0,self.defvals[2])

        self.vert_strat = IntVar()
        Checkbutton(master, text="Vertical Stratification", variable=self.vert_strat, command=self.greyout).grid(row=3)
        self.vert_strat.set(self.defvals[3])

        self.occ_h_label = Label(master, text="Occupied Height:", state=DISABLED)
        self.occ_h_label.grid(row=4)
        self.occ_h = Entry(master)
        self.occ_h.insert(0, self.defvals[4])
        self.occ_h.config(state=DISABLED)
        self.occ_h.grid(row=4, column=1)

        self.relax_assummp = IntVar()
        Checkbutton(master, text="Relax Assumptions", variable=self.relax_assummp).grid(row=5)
        self.relax_assummp.set(self.defvals[5])


    def apply(self):
        # when OK is pressed
        try:
            self.vals = [float(self.wind_f.get()),float(self.min_vent.get()),float(self.occ_gain.get()),
                         self.vert_strat.get(),float(self.occ_h.get()), self.relax_assummp.get()]
        except:
            self.initial_focus.focus_set()

    def greyout(self):
        # greys out some stuff

        if self.vert_strat.get():
            self.occ_h_label.config(state=NORMAL)
            self.occ_h.config(state=NORMAL)

        else:
            self.occ_h_label.config(state=DISABLED)
            self.occ_h.config(state=DISABLED)


if __name__ == '__main__':
    # main loop

    root = Tk()
    RoomBuilder(root)
    root.mainloop()
