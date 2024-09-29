#!/usr/bin/env python3

import radical.pilot as rp



# Task state callback to print task status updates
def task_state_cb(task, state):
    print('  task %-30s: %s' % (task.uid, task.state))


# Define the model training task
@rp.pythontask
def train_nbeats():
    import time
    from neuralforecast import NeuralForecast
    from neuralforecast.models import NBEATS
    from neuralforecast.utils import AirPassengersDF
    try:
        nf = NeuralForecast(
            models=[NBEATS(input_size=24, h=12, max_steps=100)],
            freq='M'
        )
        start_time = time.time()
        nf.fit(df=AirPassengersDF)
        nf.predict()
        end_time = time.time()

        task_time = end_time - start_time
        print(f"NBEATS model training completed in {task_time:.2f} seconds")
        return str(task_time)
    except Exception as e:
        print(f"Error during NBEATS model training: {str(e)}")
        return str(e)


if __name__ == '__main__':

    session = rp.Session()
    try:
        # Pilot description
        pd = rp.PilotDescription()
        pd.resource = 'local.localhost'
        pd.runtime = 60  # pilot runtime (minutes)
        pd.cores = 128  # number of cores

        # Initialize PilotManager and TaskManager
        pmgr = rp.PilotManager(session=session)
        tmgr = rp.TaskManager(session=session)
        tmgr.register_callback(task_state_cb)

        # Submit the pilot
        pilot = pmgr.submit_pilots([pd])[0]
        tmgr.add_pilots(pilot)

        # Set up RAPTOR master and workers
        raptor = pilot.submit_raptors(rp.TaskDescription({'mode': rp.RAPTOR_MASTER}))[0]
        workers = raptor.submit_workers(rp.TaskDescription({'mode': rp.RAPTOR_WORKER}))

        # Define 10 identical tasks for NBEATS model training
        tasks = []
        for i in range(10):
            td_nbeats = rp.TaskDescription({
                'mode': rp.TASK_FUNCTION,
                'function': train_nbeats(),
            })
            tasks.append(td_nbeats)

        # Submit tasks to RAPTOR
        submitted_tasks = raptor.submit_tasks(tasks)

        # Wait for tasks to complete
        tmgr.wait_tasks([task.uid for task in submitted_tasks])

        # Print task output (model training times)
        for task in submitted_tasks:
            print('%s [%s]: Task completed in %s seconds' % (task.uid, task.state, task.return_value))

        # Stop the RAPTOR master and wait for it to complete
        raptor.rpc('stop')
        tmgr.wait_tasks(raptor.uid)
        print('%s [%s]: %s' % (raptor.uid, raptor.state, raptor.stdout))

    finally:
        # Clean up the session
        session.close(download=False)
