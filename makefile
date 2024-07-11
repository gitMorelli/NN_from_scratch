# Define the compiler and flags
CC = gcc
PKG_CONFIG = pkg-config
PKG_FLAGS = allegro-5 allegro_font-5 allegro_primitives-5
CFLAGS = $(shell $(PKG_CONFIG) --cflags $(PKG_FLAGS))
LIBS = $(shell $(PKG_CONFIG) --libs $(PKG_FLAGS)) -lm

# Define the target executable and source file
TARGET = test
SRC = test.c

# Default rule to build the target
all: $(TARGET)

# Rule to compile the source file
$(TARGET): $(SRC)
	$(CC) $(SRC) -o $(TARGET) $(CFLAGS) $(LIBS)

# Clean up compiled files
clean:
	rm -f $(TARGET)

# Phony targets to avoid conflicts with files named 'all' or 'clean'
.PHONY: all clean

