import os
import sys
# do not use animated output in notebooks
os.environ['RADICAL_REPORT_ANIME'] = 'False'

import radical.pilot as rp
import radical.utils as ru

# determine the path of the currently active ve to simplify some examples below
ve_path = os.path.dirname(os.path.dirname(ru.which('python3')))

# create session and managers
session = rp.Session()
pmgr    = rp.PilotManager(session)
tmgr    = rp.TaskManager(session)

PWD    = os.path.abspath(os.path.dirname(__file__))


if len(sys.argv) < 2:
    cfg_file = '%s/raptor.cfg' % PWD
else:
    cfg_file = sys.argv[1]

cfg              = ru.Config(cfg=ru.read_json(cfg_file))

# submit a pilot
report = ru.Reporter(name='radical.pilot')
pd = rp.PilotDescription(cfg.pilot_descr)
pd.cores  = 1
pd.gpus   = 0
pd.runtime = 60
pilot = pmgr.submit_pilots(pd)

report.info("IM HERE")


# add the pilot to the task manager and wait for the pilot to become active
tmgr.add_pilots(pilot)
pilot.wait(rp.PMGR_ACTIVE)
report.info('pilot is up and running')

master_descr = {'mode'     : rp.RAPTOR_MASTER,
                'named_env': 'rp'}
worker_descr = {'mode'     : rp.RAPTOR_WORKER,
                'named_env': 'rp'}

report.info('Made it 1')
raptor  = pilot.submit_raptors( [rp.TaskDescription(master_descr)])[0]
report.info('Made it 2')
workers = raptor.submit_workers([rp.TaskDescription(worker_descr),
                                 rp.TaskDescription(worker_descr)])

# function for raptor to execute
@rp.pythontask
def msg(val: int):
    if(val %2 == 0):
        print('Regular message')
    else:
        print(f'This is a very odd message: {val}')

# create a minimal function task
td   = rp.TaskDescription({'mode'    : rp.TASK_FUNCTION,
                           'function': msg(3)})
task = raptor.submit_tasks([td])[0]

report.info('Made it 3')

tmgr.wait_tasks([task.uid])
report.info('Made it 4')

report.info('id: %s [%s]:\n    out:\n%s\n    ret: %s\n'
     % (task.uid, task.state, task.stdout, task.return_value))