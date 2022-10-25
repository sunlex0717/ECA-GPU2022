TARGET = arm_img_proc.app
CC = nvcc
HEADERS = $(wildcard *.h)
SOURCES = $(wildcard *.cpp )
OBJECTS = $(SOURCES:.c=.o)

all $(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) -lm

clean:
	rm -f *.bmp arm_img_proc
