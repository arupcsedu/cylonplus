#!/usr/bin/env python3

import os
import sys
import radical.pilot as rp
import radical.utils as ru

PWD = os.path.abspath(os.path.dirname(__file__))

MODEL_DIR = os.path.abspath('%s/../../model' % PWD)
CONFIG_DIR = os.path.abspath('%s/../' % PWD)

# ------------------------------------------------------------------------------
#
if __name__ == '__main__':

    # we use a reporter class for nicer output
    report = ru.Reporter(name='radical.pilot')
    report.title('MNIST CNN Training (RP version %s)' % rp.version)

    # use the resource specified as argument, fall back to localhost
    if   len(sys.argv)  > 2: report.exit('Usage:\t%s [resource]\n\n' % sys.argv[0])
    elif len(sys.argv) == 2: resource = sys.argv[1]
    else                   : resource = 'local.localhost'

    # Create a new session
    session = rp.Session()

    try:
        # read the config used for resource details
        config = ru.read_json('%s/config.json' % CONFIG_DIR)

        # Add a Pilot Manager
        pmgr = rp.PilotManager(session=session)

        # Define a compute pilot
        pd_init = {
                   'resource'      : resource,
                   'runtime'       : 30,  # pilot runtime (min)
                   'exit_on_error' : True,
                   'project'       : config[resource].get('project', None),
                   'queue'         : config[resource].get('queue',   None),
                   'access_schema' : config[resource].get('schema',  None),
                   'cores'         : config[resource].get('cores',   1),
                   'gpus'          : 1,  # Request 1 GPU
                  }
        pdesc = rp.PilotDescription(pd_init)

        # Launch the pilot
        pilot = pmgr.submit_pilots(pdesc)

        # Register the Pilot in a TaskManager object
        tmgr = rp.TaskManager(session=session)
        tmgr.add_pilots(pilot)

        # Create a task description for our ML training
        td = rp.TaskDescription()
        td.executable    = '/bin/sh'
        td.arguments     = ['single-gpu-cnn.sh']
        td.input_staging    = ['%s/single-gpu-cnn.sh' % PWD, '%s/single-gpu-cnn.py' % MODEL_DIR]
        td.ranks          = 1
        td.cores_per_rank = 8 
        td.gpus_per_rank  = 1
        td.threading_type = rp.OpenMP

        # Submit the task to the pilot
        task = tmgr.submit_tasks(td)

        # Wait for the task to complete
        tmgr.wait_tasks()

        # Print some information about the task
        report.info('Task completed with exit code: %s' % task.exit_code)
        report.info('Task stdout:\n%s' % task.stdout)
        report.info('Task stderr:\n%s' % task.stderr)

    except Exception as e:
        # Something unexpected happened in the pilot code above
        report.error('caught Exception: %s\n' % e)
        raise

    finally:
        # always clean up the session, no matter if we caught an exception or not
        report.header('finalize')
        session.close()

    report.header()