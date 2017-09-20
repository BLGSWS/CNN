obj = main.o CNN.o Layers.o Matrix.o Input.o
cnn: $(obj)
	g++ -g $(obj) -o cnn
main.o: CNN.h Input.h main.cpp
	g++ -g -c main.cpp -o main.o
CNN.o: Layers.h Matrix.h CNN.h CNN.cpp
	g++ -g -c CNN.cpp -o CNN.o
Layers.o: Matrix.h Layers.h Layers.cpp
	g++ -g -c Layers.cpp -o Layers.o
Matrix.o: Matrix.h Matrix.cpp
	g++ -g -c Matrix.cpp -o Matrix.o
Input.o: Input.h Input.cpp
	g++ -g -c Input.cpp -o Input.o
.PHONY: clean
clean:
	rm -f $(obj)