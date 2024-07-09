#!/usr/bin/env python3

'''
Demonstrate the "raptor" features for remote Task management.

This script and its supporting files may use relative file paths. Run from the
directory in which you found it.

Refer to the ``raptor.cfg`` file in the same directory for configurable run time
details.

By default, this example uses the ``local.localhost`` resource with the
``local`` access scheme where RP oversubscribes resources to emulate multiple
nodes.

In this example, we
  - Launch one or more raptor "master" task(s), which self-submits additional
    tasks (results are logged in the master's `result_cb` callback).
  - Stage scripts to be used by a raptor "Worker"
  - Provision a Python virtual environment with
    :py:func:`~radical.pilot.prepare_env`
  - Submit several tasks that will be routed through the master(s) to the
    worker(s).
  - Submit a non-raptor task in the same Pilot environment

'''

import os
import sys

import radical.utils as ru
import radical.pilot as rp


# To enable logging, some environment variables need to be set.
logger = ru.Logger('raptor')
PWD    = os.path.abspath(os.path.dirname(__file__))
RANKS  = 398


# ------------------------------------------------------------------------------
#
@rp.pythontask
def hello_world():
    print("Hello, World!")


# ------------------------------------------------------------------------------
#
def task_state_cb(task, state):
    logger.info('task %s: %s', task.uid, state)
    if state == rp.FAILED:
        logger.error('task %s failed', task.uid)


# ------------------------------------------------------------------------------
#
if __name__ == '__main__':

    if len(sys.argv) < 2:
        cfg_file = '%s/raptor.cfg' % PWD
    else:
        cfg_file = sys.argv[1]

    cfg              = ru.Config(cfg=ru.read_json(cfg_file))

    cores_per_node   = cfg.cores_per_node
    gpus_per_node    = cfg.gpus_per_node
    n_masters        = cfg.n_masters
    n_workers        = cfg.n_workers
    masters_per_node = cfg.masters_per_node
    nodes_per_worker = cfg.nodes_per_worker

    # we use a reporter class for nicer output
    report = ru.Reporter(name='radical.pilot')
    report.title('Raptor example (RP version %s)' % rp.version)

    session = rp.Session()
    try:
        pd = rp.PilotDescription(cfg.pilot_descr)
        pd.cores  = 1
        pd.gpus   = 0
        pd.runtime = 60

        pmgr = rp.PilotManager(session=session)
        tmgr = rp.TaskManager(session=session)
        tmgr.register_callback(task_state_cb)

        pilot = pmgr.submit_pilots(pd)
        tmgr.add_pilots(pilot)

        pmgr.wait_pilots(uids=pilot.uid, state=[rp.PMGR_ACTIVE])

        report.info('Stage files for the worker `hello_world` command.\n')
        pilot.stage_in({'source': ru.which('radical-pilot-hello.sh'),
                        'target': 'radical-pilot-hello.sh',
                        'action': rp.TRANSFER})

        report.info('Call pilot.prepare_env()... ')
        pilot.prepare_env(env_name='rp_dl',
                          env_spec={'type' : 'venv',
                                    'path' : '/scratch/upy9gr/workdir/rp_dl',
                                    'setup': []})
        report.info('done\n')

        master_ids = [ru.generate_id('master.%(item_counter)06d',
                                     ru.ID_CUSTOM, ns=session.uid)
                      for _ in range(n_masters)]

        tds = list()
        report.info("NUMBER OF MASTERS: "+ str(n_masters))
        for i in range(n_masters):
            td = rp.TaskDescription(cfg.master_descr)
            td.mode           = rp.RAPTOR_MASTER
            td.uid            = master_ids[i]
            td.arguments      = [cfg_file, i]
            td.cpu_processes  = 1
            td.cpu_threads    = 1
            td.named_env      = 'rp'
            td.input_staging  = [{'source': '%s/raptor_master.py' % PWD,
                                  'target': 'raptor_master.py',
                                  'action': rp.TRANSFER,
                                  'flags' : rp.DEFAULT_FLAGS},
                                 {'source': '%s/raptor_worker.py' % PWD,
                                  'target': 'raptor_worker.py',
                                  'action': rp.TRANSFER,
                                  'flags' : rp.DEFAULT_FLAGS},
                                 {'source': cfg_file,
                                  'target': os.path.basename(cfg_file),
                                  'action': rp.TRANSFER,
                                  'flags' : rp.DEFAULT_FLAGS}
                                ]
            td.pre_exec       = ['chmod +x raptor_master.py', 'chmod +x raptor_worker.py', f'chmod +x {os.path.basename(cfg_file)}']
            tds.append(td)

        if len(tds) > 0:
            report.info('Submit raptor master(s) %s\n'
                       % str([t.uid for t in tds]))
            task  = tmgr.submit_tasks(tds)
            if not isinstance(task, list):
                task = [task]

            states = tmgr.wait_tasks(
                uids=[t.uid for t in task],
                state=rp.FINAL + [rp.AGENT_EXECUTING],
                timeout=60
            )
            report.info('Master states: %s ' + str(states))
            logger.info('Master states: %s', str(states))

        tds = list()
        for i in range(1):
            tds.append(rp.TaskDescription({
                'uid'             : 'task.hello_world.%06d' % i,
                'mode'            : rp.TASK_FUNC,
                #  'ranks'           : 1,
                'function'        : hello_world(),
                'raptor_id'       : master_ids[i % n_masters]}))

        if len(tds) > 0:
            report.info('Submit tasks %s.\n' % str([t.uid for t in tds]))
            tasks = tmgr.submit_tasks(tds)
            report.info('Tasks '+ str(tasks))

            logger.info('Wait for tasks %s', [t.uid for t in tds])
            tmgr.wait_tasks(uids=[t.uid for t in tasks])

            for task in tasks:
                report.info('id: %s [%s]:\n    out: %s\n    ret: %s\n'
                      % (task.uid, task.state, task.stdout, task.return_value))

    finally:
        session.close(download=True)

    report.info('Logs from the master task should now be in local files \n')
    report.info('like %s/%s/%s.log\n' % (session.uid, pilot.uid, master_ids[0]))

# ------------------------------------------------------------------------------
