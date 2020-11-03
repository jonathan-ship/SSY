# SSY

## Table of contents

+ [General info](#general-info)
+ [Requirements](#requirements)
+ [Additional Program Requirement](#additional-program-requirement)
+ [How to Use](#how-to-use)



## General info

The Program __SSY__ is the __Stock Locating Algorithm of Steel Stock Yard Using A3C Reinforcement Learning__



##  Requirements

This module requires the following modules:

+ python=3.5(3.5.6)
+ tensorflow-gpu==1.14.0 (install gpu version of tensorflow module)
+ tensorflow==1.14.0  (install cpu version of tensorflow module)
+ scipy==1.2.1
+ pygame
+ moviepy
+ numpy==1.18.5
+ pandas
+ matplotlib



## Additional Program Requirement

In order to generate __gif__ file, __ImageMagik__ program is also required.

[ImageMagik](https://www.imagemagick.org/script/index.php)

If you are using __window OS__ , 

you should do some additional works follows...

In __config_defaults.py__ , which has a directory :  

C:\Users\user\Anaconda3\envs\\'virtual_env_name'\Lib\site-packages\moviepy , change the code

```python
IMAGEMAGICK_BINARY = os.getenv('IMAGEMAGICK_BINARY', 'auto-detect')
```

into

```python
IMAGEMAGICK_BINARY = os.getenv('IMAGEMAGICK_BINARY', 'C:\Program Files\ImageMagick-7.0.9-Q16\magick.exe')
```

`ImageMagick-7.0.9-Q16` is the ImageMagik version you had installed.

----------



## How to Use

#### Plate Configurations

+ __Assigned plates vs Random plates vs External file plates__

  This part is to decide whether you use __same plates list(Assigned plates)__ or __different plates list(Random plates)__ for each episode. If both are not the case you could use __external file__ to import import plates

  ------------

  

  + __Assigned plates__

    For __Assigned plates__ you would use same plates for every episodes. So, in __train.py__ > <class>__worker__  > member function __work()__

    `s = self.env.reset()`

    

    In __train.py__ > __main__ function, 

    the parameter should be set as `inbound_plates=inbounds` for `<class> Locating`

    ```python
    locating = Locating(max_stack=max_stack, num_pile=num_pile, inbound_plates=inbounds, observe_inbounds=observe_inbounds, display_env=False)
    ```

    + Configuration for __number of plates__

      In __train.py__ > __main__ function

      __number of plates__ could be changed by assigning different numbers for parameter `num_plate` of `generate_schedule()`

      ```python
      inbounds = generate_schedule(num_plate=50)
      ```

    + Configuration for __Random Shuffle__

      In __steelstockyard.py__ > <class>__Locating__ > __reset()__

      For __Assigned plates__ to use __Random Shuffle__ of plates list for every episodes, you should add  code line`random.shuffle(self.inbound_plates)`

      ```python
      else:
          self.inbound_plates = self.inbound_clone[(episode-1) % len(self.inbound_clone)][:]
          random.shuffle(self.inbound_plates)
      ```

    -------------------

    

  + __Random Plates__

    For __Random plates__ you would use different plates list for every episodes. 

    So, in __train.py__ > <class>__worker__  > member function __work()__

    `s = self.env.reset(hold=False)`

    

    In __train.py__ > __main__ function, 

    the parameter should be set as `inbound_plates=None  ` for <class> Locating

    ```python
    locating = Locating(max_stack=max_stack, num_pile=num_pile, inbound_plates=None, observe_inbounds=observe_inbounds, display_env=False)
    ```

    + Configuration for __number of plates__

      In __steelstockyard.py__ > <class> __Locating__ > __ __init__ __()

      __number of plates__ could be changed by assigning different numbers for parameter `num_plate` of `plate.generate_schedule()`

      ```python
      else:
          self.inbound_plates = plate.generate_schedule(250) # in this case, number of plates is 250
          self.inbound_clone = self.inbound_plates[:]
      ```

      In __steelstockyard.py__ > <class> __Locating__ > __reset()__

      You should also assign the same number for parameter `num_plate` of `plate.generate_schedule()`

      ```python
      if not hold:
          self.inbound_plates = plate.generate_schedule(250) # in this case, number of plates is 250
          self.inbound_clone = self.inbound_plates[:]
      ```

      

    + Configuration for __Random Shuffle__

      In __steelstockyard.py__ > <class>__Locating__ > __reset()__

      As you can see in the code, for __Random plates__ , __Random Shuffle__ is __meaningless__ because we generate different plates list for each episode

      ```python
      if not hold:
          self.inbound_plates = plate.generate_schedule()
          self.inbound_clone = self.inbound_plates[:]
      ```

    ----------------

    

  + __External file plates__

    In __train.py__ > __main__ function

    use `import_plates_schedule_by_week('file_dir/file_name.csv')` instead of `generate_schedule()`

    ```python
    inbounds = import_plates_schedule_by_week('../../environment/data/SampleData.csv')
    ```

    ```python
    locating = Locating(max_stack=max_stack, num_pile=num_pile, inbound_plates=inbounds, observe_inbounds = observe_inbounds, display_env=False)
    ```

    + Configuration for __Random Shuffle__

      In __steelstockyard.py__ > <class>__Locating__ > __reset()__

      For __External file plates__ to use __Random Shuffle__ of plates list for every episodes, you should add  code line`random.shuffle(self.inbound_plates)`

      ```python
      else:
          self.inbound_plates = self.inbound_clone[(episode-1) % len(self.inbound_clone)][:]
          random.shuffle(self.inbound_plates)
      ```

    + __External file Format__ (ex)

      You can see the samples data in __environment__ > __data__ > __SampleData.csv__

  ---------------

  

  #### Learning Parameters Configurations

  + __Learning Rate__

    In __train.py__ > __main__ function

    ```python
    trainer = tf.train.AdamOptimizer(learning_rate=5e-5)
    ```

    A recommanded figure for learning rate is __5e-5__ 

  + __Discount Rate__

    In __train.py__ > __main__ function

    ```python
    gamma = .99 # discount rate for advantage estimation and reward discounting
    ```

    A recommanded figure for discount rate is __0.99__

  + __Frequency of target model update(N-step bootstrapping)__

    In __A3C__ Algorithm, _each of_ __Workers__ collects samples  with its own environments.  After a __certain number of time-steps__ , target network(global network) is updated with that samples.

    In __train.py__ > __main__ function

    ```python
    if len(episode_buffer) == 30 and d != True and episode_step_count != max_episode_length-1: # in this case, frequency of target model update is 30 time-steps
    ```

  + Number of Threads

    You can also set up number of threads by changing the number of __Workers__

    In __train.py__ > __main__ function

    ```python
    num_workers = nultiprocessing.cpu.count() # Set workers to number of available CPU threads
    if num_workers > 8:
        num_workers = 8 
    workers = []
    ```

    
