cuda:
	nvcc -O2 -o gpu.app main.cu image_processing_functions_gpu.cu image_processing_functions.cpp 

cpu:
	g++ -O2 -o cpu.app img_proc.cpp image_processing_functions.cpp 

clean:
	rm -f *.bmp *.app

