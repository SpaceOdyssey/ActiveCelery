CC = g++
CXXFLAGS = -std=c++14 -O3 -march=native -DWITH_LAPACK -DNDEBUG -DEIGEN_NO_DEBUG -Wall -Wextra -pedantic -I /opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3 -I /opt/homebrew/Cellar/boost/1.86.0/include/
LIBS = -ldnest4 -lpthread -llapack -lblas
CELERITE_INCLUDE_PATH = ../celerite/cpp/include/

default:
	make noexamples -C ../DNest4/code
	$(CC) -I .. -I$(CELERITE_INCLUDE_PATH) $(CXXFLAGS) -c *.cpp
	$(CC) -pthread -L ../DNest4/code -o Celery *.o $(LIBS)
	rm *.o
