#!/usr/bin/env python3

import os
import sys
import radical.pilot as rp
import radical.utils as ru

PWD = os.path.abspath(os.path.dirname(__file__))
CONFIG_DIR = os.path.abspath('%s/../' % PWD)

if __name__ == '__main__':
    report = ru.Reporter(name='radical.pilot')
    report.title('Time Series Forecasting (RP version %s)' % rp.version)

    if   len(sys.argv)  > 2: report.exit('Usage:\t%s [resource]\n\n' % sys.argv[0])
    elif len(sys.argv) == 2: resource = sys.argv[1]
    else                   : resource = 'local.localhost'

    session = rp.Session()

    try:
        config = ru.read_json('%s/config.json' % CONFIG_DIR)

        pmgr = rp.PilotManager(session=session)

        pd_init = {
            'resource'      : resource,
            'runtime'       : 30,  # pilot runtime (min)
            'exit_on_error' : True,
            'project'       : config[resource].get('project', None),
            'queue'         : config[resource].get('queue',   None),
            'access_schema' : config[resource].get('schema',  None),
            'cores'         : config[resource].get('cores',   1),
        }
        pdesc = rp.PilotDescription(pd_init)

        pilot = pmgr.submit_pilots(pdesc)

        tmgr = rp.TaskManager(session=session)
        tmgr.add_pilots(pilot)

        td = rp.TaskDescription()
        td.executable    = 'python3'
        td.arguments     = ['time-series-forecast.py']
        td.input_staging = ['%s/time-series-forecast.py' % PWD]
        td.ranks         = 1
        td.cores_per_rank = 1

        task = tmgr.submit_tasks(td)

        tmgr.wait_tasks()

        report.info('Task completed with exit code: %s' % task.exit_code)
        report.info('Task stdout:\n%s' % task.stdout)
        report.info('Task stderr:\n%s' % task.stderr)

    except Exception as e:
        report.error('caught Exception: %s\n' % e)
        raise

    finally:
        report.header('finalize')
        session.close()

    report.header()