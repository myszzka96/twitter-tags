# Twitter Tags

To run ```rosrun <package name> <script name> ```
## ROS Service

In order to run the service, you'll need the virtual env. The order is

```bash
$ source env/bin/activate
(env) $ source ~/rasa_ws/devel/setup.bash
(env) $ rosrun slang translator.py
```

This ensures that you're falling back on the right packages. Thanks python.

### Non-Python Dependencies
```bash
$ sudo apt install mpg123
```

### Add this line to your .bashrc
```
export GOOGLE_APPLICATION_CREDENTIALS='<path_to_google_cloud_texttospeech_credentials.json>'
```

