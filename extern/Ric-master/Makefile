

CFLAGS=-DFINTEGER=int -fPIC -Wall -g -O3 -msse4 
LDFLAGS=-fPIC -Wall -g -O3 -msse4 -ljpeg -lpng 

ricflow: main.o RIC.o SLIC.o callRicflow.o
	g++ -o $@ $^  $(shell pkg-config --libs opencv) $(LDFLAGS)   
	

%.o:./%.cpp
	g++ -c $< -I./include -I/usr/local/include/ -w -fpermissive -std=c++11 $(CFLAGS) 

clean:
	rm main.o RIC.o SLIC.o callRicflow.o


