# This class/function measures time.

class time_cal:

    def __init__(self, func):
        self.func = func
    
    def __call__(self, *args, **kwargs):

        from datetime import timedelta
        import time

        start = time.time()

        self.func(*args, **kwargs)

        finish = (time.time() - start)
        
        result = self.func(*args, **kwargs)

        print(self.func.__name__,"function it takes " 
        + str(timedelta(seconds = finish)) 
        +  "\n"
        + str(timedelta(seconds = finish))[0]    + " hr " 
        + str(timedelta(seconds = finish))[2:4]  + " min "
        + str(timedelta(seconds = finish))[5:7]  + " sec "
        + str(timedelta(seconds = finish))[8:11] + " ms "
        )

        return result
        
        
        # EXAMPLE
        @time_cal
        def multi(a:int,b:int) -> int:
          return a * b
