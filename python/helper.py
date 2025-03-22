import time


class Timer:
    def __enter__(self):
        self.start=time.time()
        return self
    def __exit__(self,exc_type,exc_value,traceback):
        self.end=time.time()
        print(f'exceution Time : {self.end-self.start:.4f} sec')