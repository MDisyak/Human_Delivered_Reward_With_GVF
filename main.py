#Code is heavily modified from an initial file provided by Patrick Pilarski (BLINC Lab) at the University of Alberta
#Author: Michael Disyak
#April 9, 2017


import myLib as myLib
import threading
import random

random.seed(0)
import GVF
import RLtoolkit.tiles as tile
import numpy
import copy

import datetime
import ObservationManager
import signal
import termios, fcntl, sys, os
import ACRLPlotter
import time
import types
from lib_robotis_hack import *

import ACRL


behaviourUp = True

def getActionFromDesiredBehaviour(currentAngle):
    global behaviourUp
    if currentAngle > 1.5:
        behaviourUp = False
    elif currentAngle < -1.5:
        behaviourUp = True
    if behaviourUp:
        return 0.4
    else:
        return -0.4



# === Keyboard capture routine ===
# From: http://stackoverflow.com/questions/13207678/whats-the-simplest-way-of-detecting-keyboard-input-in-python-from-the-terminal
# BUG: on ctrl-C, termi
# nal text capture may be disabled after break
def get_char_keyboard_nonblock():
    fd = sys.stdin.fileno()
    oldterm = termios.tcgetattr(fd)
    newattr = termios.tcgetattr(fd)
    newattr[3] = newattr[3] & ~termios.ICANON & ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSANOW, newattr)
    oldflags = fcntl.fcntl(fd, fcntl.F_GETFL)
    fcntl.fcntl(fd, fcntl.F_SETFL, oldflags | os.O_NONBLOCK)
    c = None
    try:
        c = sys.stdin.read(1)
    except IOError:
        pass
    termios.tcsetattr(fd, termios.TCSAFLUSH, oldterm)
    fcntl.fcntl(fd, fcntl.F_SETFL, oldflags)
    return c


# === Graceful exit interrupt ===
def sigint_handler(signum, frame):
    print('Exiting gracefully.')
    sys.exit(0)


signal.signal(signal.SIGINT, sigint_handler)





# Method to protect robot joint limits (template to be extended based on your own robot)
def safety_check(a):
    if a < -1.75:
        return -1.75
    if a > 1.75:
        return 1.75
    return a


# === Init Variables and Learners ===
vsize = 30  # vector size (number of bins) for the state in each dimension
numactors = 2  # number of control learners
numactions = 2  # number of discrete actions (if applicable)
numsigs = 6  # numbers of other signals to be plotted
gamma = 1.0  # using average reward setting
alpha = 0.05  # try 0.05 for e-greedy, 0.5 for softmax
lamb = 0.4  # conservative due to accumulating traces
xt = numpy.array(numpy.zeros((vsize, vsize)))
xtp1 = numpy.array(numpy.array((vsize, vsize)))
R = 0  # Reward
Ravg = 0  # Average Reward
angle1 = angle2 = 0  # Observation/state signals
action1 = action2 = 0  # Action
action1new = action2new = 0  # Action
mean1 = mean2 = 0  # Policy params (means) [Cont. ACRL.py]
sigma1 = sigma2 = 0  # Policy params (sigmas) [Cont. ACRL.py]
probs1 = probs2 = [0] * numactions  # Policy params (props/values) [Disc. ACRL.py/Sarsa]
tderr1 = tderr2 = 0  # TD Error values
toggle = False

D = USB2Dynamixel_Device(dev_name="/dev/tty.usbserial-AI03QEMU", baudrate=1000000)
s_list = find_servos(D)
s1 = Robotis_Servo(D, s_list[0])
s2 = Robotis_Servo(D, s_list[1])

obsMan = ObservationManager.ObservationManager([s1,s2])
plotter = ACRLPlotter.ACRLPlotter(numactions, numactors, numsigs, vsize)

# Create two SARSA control learners
control1 = ACRL.ACRL(continuous=True, gamma=gamma, lamb=lamb, alpha=alpha, size=vsize * vsize, numactions=numactions)
control2 = ACRL.ACRL(continuous=True, gamma=gamma, lamb=lamb, alpha=alpha, size=vsize * vsize, numactions=numactions)#Sarsa.Sarsa(gamma=gamma, lamb=lamb, alpha=alpha, size=vsize * vsize, numactions=numactions)
gvf = GVF.GenValFunc(vsize, vsize*vsize)


# === Main Loop ===
print("Starting Main Loop ...")
s1.move_angle(0)
s2.move_angle(0)

for i in range(0, 20000):
   # time.sleep(.5)
    # Check for keyboard pause and other commands
    key = get_char_keyboard_nonblock()
    if not key is None:
      if key == "x":
         print("Paused.")
         while get_char_keyboard_nonblock() is None:
            pass
         print("Unpaused.")
         #toggle GVF as controller
      elif key == "p":
        print("Switching to GVF")
        if gvf.alpha == 0:
            gvf.alpha = 0.1
        else:
            gvf.alpha = 0
            #R = gvf.prediction
      else: #record reward

         if int(key) == 0:
             R = 5
         elif int(key) <= 5:
             R = -int(key)
         else:
             R = int(key) - 5
    else:
        R = 0
    start = time.time()
    print('GVF alpha is:' + str(gvf.alpha))
    # ==== States and actions resolve here ====
    # Choose action Atp1 from policy
   # action1 = getActionFromDesiredBehaviour(angle1)
    #action1 = control1.getActionContinuous()
    action2 = control2.getActionContinuous(xt.flatten())

    # Apply action via angular control command

  #  newangle1 = safety_check(action1 * 0.1 + angle1)
    newangle2 = safety_check(action2 + angle2)
    #newangle1 = safety_check(action1)
    #newangle2 = safety_check(action2)
    print ('new angle is: ' + str(newangle2))

 #   s1.move_angle(newangle1, blocking=True)     # to make it harder to learn, blocking = False!
    s2.move_angle(newangle2, blocking=True)

    # Observe next state
    angle1 = s1.read_angle()
    angle2 = s2.read_angle()

    # Set signal vector
    sigs_t = numpy.zeros(numsigs)
    sigs_t[0] = angle1
    sigs_t[1] = angle2
    sigs_t[2] = action1
    sigs_t[3] = action2
    sigs_t[4] = 0
    sigs_t[5] = 0



   # R = -abs(angle1 - angle2)
   # if angle2 > -0.1 and angle2 < 0.1:
    #    R = 0
    #else:
     #   R = -abs(0.0 - angle2)  # - abs(0.0-angle1)
  #  if angle2 > 1.15:
  #      R = 1
  #  else:
  #      R = 0

    # Turn observation into a 2D tabular state feature vector   CRAPPY TILECODER
    xtp1 = numpy.zeros((vsize, vsize))
    xtp1[int(((angle1+2.1)/4)*29), int(((angle2+2.1)/4)*29)] = 1
    #xtp1[int((angle1 + 0.51) * 19), int((angle2 + 0.51) * 19)] = 1



    # ==== Learning happens here ====

    gvf.update(xtp1.flatten(), xt.flatten(), R, 0.9, 0.9, None)
    gvf_pred = gvf.prediction

    # Get reward
    if gvf.alpha == 0: #If gvf is in control
        R = gvf.prediction

    # Update the control learner
    dstart = time.time()
    tderr1 = 0#control1.updateContinuous(xt.flatten(),xtp1.flatten(),R,action1,gamma) #0
    tderr2 = control2.updateContinuous(xt.flatten(), xtp1.flatten(), R, action2, gamma)
    Ravg = control2.Ravg




    dend = time.time()

    # Fill these in when you make your continuous action ACRL.py learner!
    mean1 = 0#control1.getMean()
    mean2 = control2.getMean()
    sigma1 = 0#control1.getSigma()
    sigma2 = control2.getSigma()

    plotData = {'mean1': mean1, 'mean2': mean2, 'sigma1': sigma1, 'sigma2': sigma2, 'tderr1': tderr1, 'tderr2': tderr2, 'R': R,'Ravg': Ravg, 'probs1': probs1, 'probs2': probs2, 'xtp1': xtp1, 'sigs_t': sigs_t, 'gvf_pred': gvf_pred}
    plotter.plotUpdate(plotData)

    xt = xtp1

    #Calculate timings (used in plotter)
    end = time.time()
    latency = end - start
    dlatency = dend - dstart

    #Set text elements in plotter
    plotter.t2.set_text("Steptime: " + str.format('{0:.3f}', latency) + " s")
    plotter.t3.set_text("LearnerSteptime: " + str.format('{0:.3f}', dlatency) + " s")
    plotter.t4.set_text("MeanAvgReward100ts: " + str.format('{0:.3f}', plotter.buff_avgrewards[-101:].mean()))

    plotter.t5.set_text('Time Step: ' + str(i))
    plotter.t6.set_text("MeanGVFPrediction100ts: " + str.format('{0:.3f}', plotter.buff_gvfpredictions[-101:].mean()))
   # if i % 1000 == 0:
        #plotter.saveFigure()
