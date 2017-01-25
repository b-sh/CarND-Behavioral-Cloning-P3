The car will just sit there until your Python server connects to it and provides it steering angles. Here’s how you start your Python server:

    Set up your development environment with the CarND Starter Kit.
    Download drive.py.
    Run the server.
        python drive.py model.json
        If you're using Docker for this project: docker run -it --rm -p 4567:4567 -v `pwd`:/src udacity/carnd-term1-starer-kit python drive.py model.json or docker run -it --rm -p 4567:4567 -v ${pwd}:/src udacity/carnd-term1-starer-kit python drive.py model.json. Port 4567 is used by the simulator to communicate.

Once the model is up and running in drive.py, you should see the car move around (and hopefully not off) the track!
