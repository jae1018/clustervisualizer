import pandas as pd
import numpy as np
import matplotlib.pyplot as plt










class temp:
    
    """
    Simulates a temperature measurement based on position
    """
    
    def __init__(self, low_temp_kwargs, high_temp_kwargs, bnds):
        
        """
        Creates temp_sim instance
        
        Parameters
        ----------
        low_temp_kwargs: kwargs dict for np.random.normal
            Kwargs dict that will simulate low-temperature gas measurement
            
        high_temp_kwargs: kwargs dict for no.random.normal
            Kwargs dict that will simulate high-temperature gas measurement
            
        bnd: float or 2-element arr
            If float, then have a hard boundary between low and
            high temp gas.
            If 2-floats, then elements correspond to starting and
            ending boundaries. If left of starting boundary, then
            sample low temp gas; if right of ending boundary, then
            sample high temp gas; if in the middle sample both and
            take average.
        """
        
        self._lt_kwargs = low_temp_kwargs
        self._ht_kwargs = high_temp_kwargs
        
        if isinstance(bnds, (list, tuple, np.ndarray)):
            self.bnds = bnds
        else:
            self.bnds = [ bnds ]
    
    
    
    def _make_lt_measurement(self):
        
        """
        Make a low-temp measurement
        """
        
        return np.random.normal( **self._lt_kwargs )
    
    
    
    def _make_ht_measurement(self):
        
        """
        Make a high-temp measurement
        """
        
        return np.random.normal( **self._ht_kwargs )
    
    
    
    def make_measurement(self, x_posit):
        
        """
        Make simulated measurements
        """
        
        ## If only one bound, then there's a *hard* boundary
        ## so sample from low temp gas if <= bound, or sample from
        ## high temp gas if . bound
        if len(self.bnds) == 1:
            # If left of hard boundary, sample low temp gas
            if x_posit <= self.bnds[0]:
                return self._make_lt_measurement()
            # If right of hard boundary, sample high temp gas
            else:
                return self._make_ht_measurement()
        
        ## If two bounds, then there's a soft boudnary
        else:
            lowt = self._make_lt_measurement()
            hight = self._make_ht_measurement()
            # If left of start boundary, then sample low temp
            if x_posit <= self.bnds[0]:
                return lowt
            # If right of end boundary, then sample high temp
            elif x_posit > self.bnds[1]:
                return hight
            # If between boundaries, take average of low and high temp
            else:
                return (lowt + hight)/2
                
                
                
                










class detector:
    
    """
    Simulates a detector's position and velocity
    """
    
    def __init__(self,container):
        
        """
        4 consec args that are x, y, vx, vy
        container can be any container of numbers (tuple, list, numpy arr)
        """
        
        self.x = container[0]
        self.y = container[1]
        self.vx = container[2]
        self.vy = container[3]
        
        
        
    def update_posit(self, new_posit):
        
        """
        Updates position
        """
        
        self.x = new_posit[0]
        self.y = new_posit[1]
        
        
        
    def update_veloc(self, new_veloc):
        
        """
        Updates velocity
        """
        
        self.vx = new_veloc[0]
        self.vy = new_veloc[1]
        
        
        
    def get_posit(self):
        
        """
        Retrieves position as a 2-elem numpy array
        """
        
        return np.array( [self.x, self.y] )
    
    
    
    def get_veloc(self):
        
        """
        Retrieves velocity as a 2-elem numpy array
        """
        
        return np.array( [self.vx, self.vy] )
    
    
    
    def move(self, box_inst):
        
        """
        Move the detector according to its position and velocity.
        
        Parameters
        ---------
        box_inst: instance of box class
            Contains info related to boundaries of detector
            
        Returns
        -------
        None
        """
        
        ## Update posit and save arr copy of veloc
        posit = self.get_posit()
        veloc = self.get_veloc()
        new_posit = posit + veloc
        new_veloc = veloc.copy()
        
        ## new_x > x2
        if new_posit[0] > box_inst.x2:
            dx_past_x2 = new_posit[0] - box_inst.x2
            new_posit[0] = box_inst.x2 - dx_past_x2
            new_veloc[0] = -1 * veloc[0]
            
        ## new_x < x1
        if new_posit[0] < box_inst.x1:
            dx_past_x1 = box_inst.x1 - new_posit[0]
            new_posit[0] = box_inst.x1 + dx_past_x1
            new_veloc[0] = -1 * veloc[0]
            
        ## new_y > y2
        if new_posit[1] > box_inst.y2:
            dy_past_y2 = new_posit[1] - box_inst.y2
            new_posit[1] = box_inst.y2 - dy_past_y2
            new_veloc[1] = -1 * veloc[1]
        
        ## new_y < y1
        if new_posit[1] < box_inst.y1:
            dy_past_y1 = box_inst.y1 - new_posit[1]
            new_posit[1] = box_inst.y1 + dy_past_y1
            new_veloc[1] = -1 * veloc[1]
            
        ## update detector position and veloc
        self.update_posit(new_posit)
        self.update_veloc(new_veloc)
        
        









class box:
    
    """
    Simulates the bounds of a box for a moving detector
    
    y2  ---------------
        |             |
        |             |
        |             |
        |             |
        |             |
    y1  ---------------
        x1            x2
        
    """
    
    def __init__(self,container):
        
        """
        4 consec args that are x1, x2, y1, y2
        container can be any container of numbers (tuple, list, numpy arr)
        """
        
        self.x1 = container[0]
        self.x2 = container[1]
        self.y1 = container[2]
        self.y2 = container[3]











class simulate_detector:
    
    """
    Simulates a temperature detector moving between low
    and high temperature gases
    
    Functions
    ---------
    TBD
    """
    
    def __init__(self,
                 init_posit    = None,
                 init_veloc    = None,
                 box_bnds      = None,
                 max_iter      = None,
                 lowt_kwargs   = None,
                 hight_kwargs  = None,
                 temp_boundary = None):
        
        """
        Init the simulation of a detector
        
        
        Parameters
        ---------
        init_posit: 2-element container
            The initial (x,y) position of the detector
            
        init_veloc: 2-element container
            The initial (vx,vy) velocity of the detector
        
        box_bnds: 4-element container
            The bounds of the box the detector is in.
            Format is (x1, x2, y1, y2)
            
        max_iter: int
            Number of iterations to move the detector
            
        lowt_kwargs: kwargs dict for np.random.normal
            Kwargs used to generate low temperature measurement
            
        hight_kwargs: kwargs dict for np.random.normal
            Kwargs used to generate high temperature measurement
        
        temp_boundary: number or 2-element container
            If single number, then transition from low temp to high temp
            gas is a hard boundary
            If two numbers, then first / second numbers are the start /
            end boundary lines; positions between start and end record
            temperature measurements that are the *average* of low and high.
        """
        
        ## Init params
        if init_posit is None: init_posit = (0.25, 0.5)
        if init_veloc is None: init_veloc = (0.125,0)
        if box_bnds is None: box_bnds = (0,1,0,1)  # x1, x2, y1, y2
        if max_iter is None: max_iter = 100
        if lowt_kwargs is None: lowt_kwargs = { "loc" : 10, 'scale' : 1 }
        if hight_kwargs is None: hight_kwargs = { "loc" : 100, 'scale' : 10 }
        if temp_boundary is None:
            temp_boundary = (box_bnds[1] - box_bnds[0])/2 + box_bnds[0]
        
        
        ## Create instances of classes
        # detector
        self.detector = detector( [*init_posit, *init_veloc] )
        # temp simulator
        self.temp = temp( lowt_kwargs,
                          hight_kwargs,
                          temp_boundary )
        # box for detector bounds
        self.box = box(box_bnds)
        
        
        ## Save other class params
        self.max_iter = max_iter
        self.posit_arr = np.zeros((max_iter,2))
        self.veloc_arr = np.zeros((max_iter,2))
        self.temp_arr = np.zeros(max_iter)
    
    
    
    def _record_data(self, n):
        
        """
        Record position, velocity, and temp data for iteration number n
        
        Parameters
        ----------
        n: int
            0 <= n < max_iter
            Which iteration is detector on
        
        Returns
        -------
        None
        """
        
        self.posit_arr[n,:] = self.detector.get_posit()
        self.veloc_arr[n,:] = self.detector.get_veloc()
        self.temp_arr[n] = self.temp.make_measurement( self.detector.x )
        
        
        
    def start(self):
        
        """
        Begin the simulation up to iteration max_iter
        
        Parameters
        ----------
        None
        
        Returns
        -------
        None
        """
        
        ## Save init data
        self._record_data(0)
        for i in range(1, self.max_iter):
            ## Move detector and make new measurements
            self.detector.move( self.box )
            self._record_data(i)
    
    
    
    def get_data(self):
        
        """
        Retrieve detector data as a pandas dataframe
        
        Parameters
        ----------
        None
        
        Returns 5-column pandas dataframe
        """
        
        return pd.DataFrame({
                    'x' : self.posit_arr[:,0],
                    'y' : self.posit_arr[:,1],
                    'vx' : self.veloc_arr[:,0],
                    'vy' : self.veloc_arr[:,1],
                    'temp' : self.temp_arr
                            })
    
    
    
    def plot_every(self, n):
        
        """
        Create a plot showing detector position and velocity
        as well as box bondaries
        """
        
        for i in range(self.max_iter):
            
            if i % n == 0:
        
                ## init figure
                fig, axes = plt.subplots(1,1)
                
                kwargs = { "linestyle" : "dashed",
                           "c" : "black" }
                
                ## setup box bounds ...
                # for x1
                axes.axvline(x    = self.box.x1,
                             ymin = self.box.y1,
                             ymax = self.box.y2,
                             **kwargs)
                # for x2
                axes.axvline(x    = self.box.x2,
                             ymin = self.box.y1,
                             ymax = self.box.y2,
                             **kwargs)
                # for y1
                axes.axhline(y    = self.box.y1,
                             xmin = self.box.x1,
                             xmax = self.box.x2,
                             **kwargs)
                # for y2
                axes.axhline(y    = self.box.y2,
                             xmin = self.box.x1,
                             xmax = self.box.x2,
                             **kwargs)
                
                ## setup x/y plot bounds based on delta window
                delta = 0.1
                # compute x bounds
                extra_x = delta * (self.box.x2 - self.box.x1)
                x_bot = self.box.x1 - extra_x/2
                x_top = self.box.x2 + extra_x/2
                # compute y bounds
                extra_y = delta * (self.box.y2 - self.box.y1)
                y_bot = self.box.y1 - extra_y/2
                y_top = self.box.y2 + extra_y/2
                # set bounds
                axes.set_xlim( [x_bot, x_top] )
                axes.set_ylim( [y_bot, y_top] )
                
                ## plot detector posit
                axes.scatter(*self.posit_arr[i,:], s=5, marker='o')
                
                ## plot arrow indicating veloc
                axes.arrow(
                    *self.posit_arr[i,:],
                    *self.veloc_arr[i,:]
                          )
                
                ## show and close
                plt.show()
                plt.close()
        
        
        
        
        
            
            
    







        
        
        
        





detector_sim = simulate_detector(
                    init_posit = (0.1,0.5),
                    init_veloc = (-0.0682,0.0135798),
                    box_bnds = (0,1,0,1),
                    max_iter = 2000
                                )


detector_sim.start()


df = detector_sim.get_data()

detector_sim.plot_every(100)

        