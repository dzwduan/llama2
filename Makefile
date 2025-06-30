# Compiler settings
CC = gcc
CFLAGS_COMMON = -Wall -Wextra
CFLAGS_RELEASE = -std=c11 $(CFLAGS_COMMON) -O3
CFLAGS_DEBUG = -std=c11 $(CFLAGS_COMMON) -g -DDEBUG
CFLAGS_MEMCHECK = -std=c11 $(CFLAGS_COMMON) -g -DDEBUG -fsanitize=address -fno-omit-frame-pointer

# Directories
SRCDIR = .
OBJDIR = obj
BINDIR = ..
OBJDIR_DEBUG = $(OBJDIR)/debug
OBJDIR_RELEASE = $(OBJDIR)/release
OBJDIR_MEMCHECK = $(OBJDIR)/memcheck

# Target executables
TARGET_RELEASE = $(BINDIR)/run
TARGET_DEBUG = $(BINDIR)/run_debug
TARGET_MEMCHECK = $(BINDIR)/run_memcheck

# Source files
SRCS = $(wildcard $(SRCDIR)/*.c)
OBJS_RELEASE = $(patsubst $(SRCDIR)/%.c, $(OBJDIR_RELEASE)/%.o, $(SRCS))
OBJS_DEBUG = $(patsubst $(SRCDIR)/%.c, $(OBJDIR_DEBUG)/%.o, $(SRCS))
OBJS_MEMCHECK = $(patsubst $(SRCDIR)/%.c, $(OBJDIR_MEMCHECK)/%.o, $(SRCS))
DEPS = $(wildcard $(SRCDIR)/*.h)

# Default target (release)
all: release

# Create directories if they don't exist
$(shell mkdir -p $(OBJDIR_RELEASE) $(OBJDIR_DEBUG) $(OBJDIR_MEMCHECK) $(BINDIR))

# Release build
release: CFLAGS = $(CFLAGS_RELEASE)
release: $(TARGET_RELEASE)

# Debug build
debug: CFLAGS = $(CFLAGS_DEBUG)
debug: $(TARGET_DEBUG)

# Memory leak detection build
memcheck: CFLAGS = $(CFLAGS_MEMCHECK)
memcheck: LDFLAGS = -fsanitize=address
memcheck: $(TARGET_MEMCHECK)

# Linking
$(TARGET_RELEASE): $(OBJS_RELEASE)
	$(CC) $(CFLAGS) $^ -o $@ -lm

$(TARGET_DEBUG): $(OBJS_DEBUG)
	$(CC) $(CFLAGS) $^ -o $@ -lm

$(TARGET_MEMCHECK): $(OBJS_MEMCHECK)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@ -lm

# Compilation
$(OBJDIR_RELEASE)/%.o: $(SRCDIR)/%.c $(DEPS)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR_DEBUG)/%.o: $(SRCDIR)/%.c $(DEPS)
	$(CC) $(CFLAGS) -c $< -o $@

$(OBJDIR_MEMCHECK)/%.o: $(SRCDIR)/%.c $(DEPS)
	$(CC) $(CFLAGS) -c $< -o $@

# Clean
clean:
	rm -rf $(OBJDIR) $(TARGET_RELEASE) $(TARGET_DEBUG) $(TARGET_MEMCHECK)

# Phony targets
.PHONY: all release debug memcheck clean

