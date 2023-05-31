#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2021.2.3),
    on May 31, 2023, at 15:54
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

from __future__ import absolute_import, division

from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard



# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)

# Store info about the experiment session
psychopyVersion = '2021.2.3'
expName = 'exp1'  # from the Builder filename that created this script
expInfo = {'participant': '', 'session': '001'}
dlg = gui.DlgFromDict(dictionary=expInfo, sortKeys=False, title=expName)
if dlg.OK == False:
    core.quit()  # user pressed cancel
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='C:\\Users\\csahi\\work & study\\probabilistic learning\\visual rewards\\exp1_lastrun.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# Setup the Window
win = visual.Window(
    size=[1280, 720], fullscr=False, screen=0, 
    winType='pyglet', allowGUI=True, allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='height')
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess

# Setup eyetracking
ioDevice = ioConfig = ioSession = ioServer = eyetracker = None

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard()

# Initialize components for Routine "task_type"
task_typeClock = core.Clock()
text_2 = visual.TextStim(win=win, name='text_2',
    text='',
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# Initialize components for Routine "trial"
trialClock = core.Clock()
key_resp = keyboard.Keyboard()
image = visual.ImageStim(
    win=win,
    name='image', 
    image='sin', mask=None,
    ori=0.0, pos=(0, 0), size=(1.8, 1.0),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)

# Initialize components for Routine "highlight_door"
highlight_doorClock = core.Clock()
image_2 = visual.ImageStim(
    win=win,
    name='image_2', 
    image='sin', mask=None,
    ori=0.0, pos=(0, 0), size=(1.8, 1.0),
    color=[1,1,1], colorSpace='rgb', opacity=1.0,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)

# Initialize components for Routine "feedback_points"
feedback_pointsClock = core.Clock()
image_3 = visual.ImageStim(
    win=win,
    name='image_3', 
    image='sin', mask=None,
    ori=0.0, pos=(0, 0), size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
text = visual.TextStim(win=win, name='text',
    text='',
    font='Open Sans',
    pos=(-0.08, -0.04), height=0.1, wrapWidth=None, ori=0.0, 
    color='black', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=-2.0);

# Initialize components for Routine "feedback_images"
feedback_imagesClock = core.Clock()
image_4 = visual.ImageStim(
    win=win,
    name='image_4', 
    image='sin', mask=None,
    ori=0.0, pos=(0, 0), size=1.0,
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-1.0)
image_5 = visual.ImageStim(
    win=win,
    name='image_5', 
    image='sin', mask=None,
    ori=0.0, pos=(0, 0), size=(0.85,0.85),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=-2.0)

# Initialize components for Routine "exp_break"
exp_breakClock = core.Clock()

# Initialize components for Routine "task_break"
task_breakClock = core.Clock()
text_3 = visual.TextStim(win=win, name='text_3',
    text='You have completed half the experiment.\n\nWell done! You can now take a break.\n\nPlease press the space bar to resume.',
    font='Open Sans',
    pos=(0, 0), height=0.1, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
key_resp_3 = keyboard.Keyboard()

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.CountdownTimer()  # to track time remaining of each (non-slip) routine 

# set up handler to look after randomisation of conditions etc
block = data.TrialHandler(nReps=1.0, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions('choose_task.xlsx'),
    seed=None, name='block')
thisExp.addLoop(block)  # add the loop to the experiment
thisBlock = block.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
if thisBlock != None:
    for paramName in thisBlock:
        exec('{} = thisBlock[paramName]'.format(paramName))

for thisBlock in block:
    currentLoop = block
    # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
    if thisBlock != None:
        for paramName in thisBlock:
            exec('{} = thisBlock[paramName]'.format(paramName))
    
    # ------Prepare to start Routine "task_type"-------
    continueRoutine = True
    routineTimer.add(1.000000)
    # update component parameters for each repeat
    text_2.setText(message
)
    # keep track of which components have finished
    task_typeComponents = [text_2]
    for thisComponent in task_typeComponents:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    task_typeClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
    frameN = -1
    
    # -------Run Routine "task_type"-------
    while continueRoutine and routineTimer.getTime() > 0:
        # get current time
        t = task_typeClock.getTime()
        tThisFlip = win.getFutureFlipTime(clock=task_typeClock)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_2* updates
        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            text_2.setAutoDraw(True)
        if text_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_2.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                text_2.tStop = t  # not accounting for scr refresh
                text_2.frameNStop = frameN  # exact frame index
                win.timeOnFlip(text_2, 'tStopRefresh')  # time at next scr refresh
                text_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in task_typeComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # -------Ending Routine "task_type"-------
    for thisComponent in task_typeComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    block.addData('text_2.started', text_2.tStartRefresh)
    block.addData('text_2.stopped', text_2.tStopRefresh)
    
    # set up handler to look after randomisation of conditions etc
    trials = data.TrialHandler(nReps=1.0, method='sequential', 
        extraInfo=expInfo, originPath=-1,
        trialList=data.importConditions(condsFile),
        seed=None, name='trials')
    thisExp.addLoop(trials)  # add the loop to the experiment
    thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
    if thisTrial != None:
        for paramName in thisTrial:
            exec('{} = thisTrial[paramName]'.format(paramName))
    
    for thisTrial in trials:
        currentLoop = trials
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                exec('{} = thisTrial[paramName]'.format(paramName))
        
        # ------Prepare to start Routine "trial"-------
        continueRoutine = True
        routineTimer.add(2.000000)
        # update component parameters for each repeat
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        image.setImage(background)
        # keep track of which components have finished
        trialComponents = [key_resp, image]
        for thisComponent in trialComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        trialClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "trial"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = trialClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=trialClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *key_resp* updates
            if key_resp.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                key_resp.status = STARTED
                # keyboard checking is just starting
                key_resp.clock.reset()  # now t=0
                key_resp.clearEvents(eventType='keyboard')
            if key_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp.tStop = t  # not accounting for scr refresh
                    key_resp.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(key_resp, 'tStopRefresh')  # time at next scr refresh
                    key_resp.status = FINISHED
            if key_resp.status == STARTED:
                theseKeys = key_resp.getKeys(keyList=['left', 'right'], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = [key.name for key in _key_resp_allKeys]  # storing all keys
                    key_resp.rt = [key.rt for key in _key_resp_allKeys]
                    # was this correct?
                    if (key_resp.keys == str(correctAns)) or (key_resp.keys == correctAns):
                        key_resp.corr = 1
                    else:
                        key_resp.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # *image* updates
            if image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image.frameNStart = frameN  # exact frame index
                image.tStart = t  # local t and not account for scr refresh
                image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image, 'tStartRefresh')  # time at next scr refresh
                image.setAutoDraw(True)
            if image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    image.tStop = t  # not accounting for scr refresh
                    image.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(image, 'tStopRefresh')  # time at next scr refresh
                    image.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trialComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "trial"-------
        for thisComponent in trialComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
            # was no response the correct answer?!
            if str(correctAns).lower() == 'none':
               key_resp.corr = 1;  # correct non-response
            else:
               key_resp.corr = 0;  # failed to respond (incorrectly)
        # store data for trials (TrialHandler)
        trials.addData('key_resp.keys',key_resp.keys)
        trials.addData('key_resp.corr', key_resp.corr)
        if key_resp.keys != None:  # we had a response
            trials.addData('key_resp.rt', key_resp.rt)
        trials.addData('image.started', image.tStartRefresh)
        trials.addData('image.stopped', image.tStopRefresh)
        
        # ------Prepare to start Routine "highlight_door"-------
        continueRoutine = True
        routineTimer.add(0.500000)
        # update component parameters for each repeat
        if key_resp.keys == ['left'] :
            highlight = highlight_left_choice
        elif key_resp.keys == ['right'] :
            highlight = highlight_right_choice
        else :
            highlight = background
        image_2.setImage(highlight)
        # keep track of which components have finished
        highlight_doorComponents = [image_2]
        for thisComponent in highlight_doorComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        highlight_doorClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "highlight_door"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = highlight_doorClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=highlight_doorClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image_2* updates
            if image_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_2.frameNStart = frameN  # exact frame index
                image_2.tStart = t  # local t and not account for scr refresh
                image_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_2, 'tStartRefresh')  # time at next scr refresh
                image_2.setAutoDraw(True)
            if image_2.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_2.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    image_2.tStop = t  # not accounting for scr refresh
                    image_2.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(image_2, 'tStopRefresh')  # time at next scr refresh
                    image_2.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in highlight_doorComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "highlight_door"-------
        for thisComponent in highlight_doorComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        trials.addData('image_2.started', image_2.tStartRefresh)
        trials.addData('image_2.stopped', image_2.tStopRefresh)
        
        # ------Prepare to start Routine "feedback_points"-------
        continueRoutine = True
        routineTimer.add(1.500000)
        # update component parameters for each repeat
        if condsFile == 'points_positive_task.csv':
        
            if not key_resp.keys :
                msg="Failed\n to respond"
            elif key_resp.corr:#stored on last run routine
                msg=positive_arm
            else:
                msg=neutral_arm
                
        elif condsFile == 'images_positive_task.csv':
            msg = "Failed"
            continueRoutine = False
            routineForceEnded = True
        image_3.setSize(background_size)
        image_3.setImage(feedback_background)
        text.setText(msg)
        # keep track of which components have finished
        feedback_pointsComponents = [image_3, text]
        for thisComponent in feedback_pointsComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        feedback_pointsClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "feedback_points"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = feedback_pointsClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=feedback_pointsClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image_3* updates
            if image_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_3.frameNStart = frameN  # exact frame index
                image_3.tStart = t  # local t and not account for scr refresh
                image_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_3, 'tStartRefresh')  # time at next scr refresh
                image_3.setAutoDraw(True)
            if image_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_3.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    image_3.tStop = t  # not accounting for scr refresh
                    image_3.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(image_3, 'tStopRefresh')  # time at next scr refresh
                    image_3.setAutoDraw(False)
            
            # *text* updates
            if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                text.setAutoDraw(True)
            if text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    text.tStop = t  # not accounting for scr refresh
                    text.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(text, 'tStopRefresh')  # time at next scr refresh
                    text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedback_pointsComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "feedback_points"-------
        for thisComponent in feedback_pointsComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        
        # ------Prepare to start Routine "feedback_images"-------
        continueRoutine = True
        routineTimer.add(1.500000)
        # update component parameters for each repeat
        if condsFile == 'images_positive_task.csv':
        
            if not key_resp.keys :
                img= 'background/frame.png'
            elif key_resp.corr:#stored on last run routine
                img=positive_arm
            else:
                img=neutral_arm
              
        elif condsFile == 'points_positive_task.csv':
            img= 'background/frame.png'
            continueRoutine = False
            routineForceEnded = True
        image_4.setSize(background_size)
        image_4.setImage(feedback_background)
        image_5.setImage(img)
        # keep track of which components have finished
        feedback_imagesComponents = [image_4, image_5]
        for thisComponent in feedback_imagesComponents:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        feedback_imagesClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
        frameN = -1
        
        # -------Run Routine "feedback_images"-------
        while continueRoutine and routineTimer.getTime() > 0:
            # get current time
            t = feedback_imagesClock.getTime()
            tThisFlip = win.getFutureFlipTime(clock=feedback_imagesClock)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image_4* updates
            if image_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_4.frameNStart = frameN  # exact frame index
                image_4.tStart = t  # local t and not account for scr refresh
                image_4.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_4, 'tStartRefresh')  # time at next scr refresh
                image_4.setAutoDraw(True)
            if image_4.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_4.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    image_4.tStop = t  # not accounting for scr refresh
                    image_4.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(image_4, 'tStopRefresh')  # time at next scr refresh
                    image_4.setAutoDraw(False)
            
            # *image_5* updates
            if image_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_5.frameNStart = frameN  # exact frame index
                image_5.tStart = t  # local t and not account for scr refresh
                image_5.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_5, 'tStartRefresh')  # time at next scr refresh
                image_5.setAutoDraw(True)
            if image_5.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_5.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    image_5.tStop = t  # not accounting for scr refresh
                    image_5.frameNStop = frameN  # exact frame index
                    win.timeOnFlip(image_5, 'tStopRefresh')  # time at next scr refresh
                    image_5.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
                core.quit()
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedback_imagesComponents:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # -------Ending Routine "feedback_images"-------
        for thisComponent in feedback_imagesComponents:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials'
    
    thisExp.nextEntry()
    
# completed 1.0 repeats of 'block'


# ------Prepare to start Routine "exp_break"-------
continueRoutine = True
# update component parameters for each repeat
if trials.thisN != 89:
    continueRoutine = False
    
# keep track of which components have finished
exp_breakComponents = []
for thisComponent in exp_breakComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
exp_breakClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "exp_break"-------
while continueRoutine:
    # get current time
    t = exp_breakClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=exp_breakClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in exp_breakComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "exp_break"-------
for thisComponent in exp_breakComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# the Routine "exp_break" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# ------Prepare to start Routine "task_break"-------
continueRoutine = True
# update component parameters for each repeat
key_resp_3.keys = []
key_resp_3.rt = []
_key_resp_3_allKeys = []
# keep track of which components have finished
task_breakComponents = [text_3, key_resp_3]
for thisComponent in task_breakComponents:
    thisComponent.tStart = None
    thisComponent.tStop = None
    thisComponent.tStartRefresh = None
    thisComponent.tStopRefresh = None
    if hasattr(thisComponent, 'status'):
        thisComponent.status = NOT_STARTED
# reset timers
t = 0
_timeToFirstFrame = win.getFutureFlipTime(clock="now")
task_breakClock.reset(-_timeToFirstFrame)  # t0 is time of first possible flip
frameN = -1

# -------Run Routine "task_break"-------
while continueRoutine:
    # get current time
    t = task_breakClock.getTime()
    tThisFlip = win.getFutureFlipTime(clock=task_breakClock)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *text_3* updates
    if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        text_3.frameNStart = frameN  # exact frame index
        text_3.tStart = t  # local t and not account for scr refresh
        text_3.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
        text_3.setAutoDraw(True)
    
    # *key_resp_3* updates
    if key_resp_3.status == NOT_STARTED and t >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        key_resp_3.frameNStart = frameN  # exact frame index
        key_resp_3.tStart = t  # local t and not account for scr refresh
        key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
        key_resp_3.status = STARTED
        # keyboard checking is just starting
        key_resp_3.clock.reset()  # now t=0
    if key_resp_3.status == STARTED:
        theseKeys = key_resp_3.getKeys(keyList=['space'], waitRelease=False)
        _key_resp_3_allKeys.extend(theseKeys)
        if len(_key_resp_3_allKeys):
            key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
            key_resp_3.rt = _key_resp_3_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in task_breakComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# -------Ending Routine "task_break"-------
for thisComponent in task_breakComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
thisExp.addData('text_3.started', text_3.tStartRefresh)
thisExp.addData('text_3.stopped', text_3.tStopRefresh)
# check responses
if key_resp_3.keys in ['', [], None]:  # No response was made
    key_resp_3.keys = None
thisExp.addData('key_resp_3.keys',key_resp_3.keys)
if key_resp_3.keys != None:  # we had a response
    thisExp.addData('key_resp_3.rt', key_resp_3.rt)
thisExp.nextEntry()
# the Routine "task_break" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
