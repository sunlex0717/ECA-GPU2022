TARGET = arm_img_proc
CFLAGS =-O0 -g -Wall
HEADERS = $(wildcard *.h)
SOURCES = $(wildcard *.c )
OBJECTS = $(SOURCES:.c=.o)

all $(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) -o $(TARGET) $(SOURCES) -lm

clean:
	rm -f $(OBJECTS) $(TARGET)
