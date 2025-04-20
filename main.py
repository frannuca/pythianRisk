from regression.regime_base import *
from contracts.projection_pb2 import RegimeBoundary

def main():
    a = RegimeBoundary()
    a.factor_name = "Hi"
    a.upper_bound = 0.5
    a.lower_bound = 0.1
    
    print(a)


if __name__ == "__main__":
    main()
