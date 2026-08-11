# 🐍 Python — Foundation to Systems

> A practical journey through Python — starting from the basics and gradually moving towards **systems programming, data, ML, and AI systems**.

This repository contains my Python learning journey, notes, experiments, and projects.

The goal isn't just to learn Python syntax.

I want to understand how Python is used in real software and AI/ML systems, what happens underneath the abstractions, and where Python connects with lower-level technologies like **C/C++, operating systems, CPUs, GPUs, and embedded systems**.

---

## 🗺️ Roadmap

```text
Python Basics
     ↓
Problem Solving
     ↓
Data Structures & Functions
     ↓
OOP & Modules
     ↓
Files / OS / Networking
     ↓
Concurrency & Python Internals
     ↓
NumPy / Pandas
     ↓
Machine Learning
     ↓
Deep Learning
     ↓
AI/ML Systems
     ↓
Performance + C/C++ + Hardware
```

---

## 📚 What's Inside

### 01. Python Fundamentals

Starting with the language itself.

* Variables & data types
* Operators
* Input / Output
* Strings
* Conditions
* Loops
* Functions
* Error handling
* Modules & packages

Some of the early programs are intentionally simple. The point is to build a solid base before jumping into libraries.

---

### 02. Data Structures & Problem Solving

Python's built-in structures:

* Lists
* Tuples
* Sets
* Dictionaries
* Stacks & Queues
* Comprehensions
* Iterators & Generators

Alongside this, I'll use Python for basic algorithmic problem solving and gradually connect it with my **DSA preparation**.

---

### 03. Object Oriented Python

Understanding how larger programs can be structured.

Topics include:

* Classes & objects
* Constructors
* Encapsulation
* Inheritance
* Polymorphism
* Composition
* Abstract classes
* Dataclasses
* Magic methods

Example systems will be built instead of only writing isolated class examples.

---

### 04. Files, Data & Automation

Python becomes much more useful once it starts interacting with the outside world.

* File handling
* `pathlib`
* CSV
* JSON
* Serialization
* Directory management
* Logging
* Environment variables
* CLI arguments

#### Small Projects

* File organizer
* Log analyzer
* CSV → JSON converter
* Configuration manager
* CLI utilities

---

# 🖥️ Python → Systems

This is where I want to move beyond normal beginner Python.

## Operating Systems

Exploring Python's interaction with the OS:

* Processes
* Environment variables
* Filesystems
* Subprocesses
* Signals
* System information
* Process monitoring

Useful modules:

```text
os
sys
pathlib
subprocess
shutil
signal
platform
```

### Project: System Monitor

A small CLI utility that can display things like:

```text
CPU Usage
Memory Usage
Disk Usage
Running Processes
Network Information
OS Information
```

The idea is to understand the relationship between:

```text
Python
  ↓
Operating System
  ↓
Processes
  ↓
System Resources
```

---

# 🌐 Networking

Python will also be used to understand networking from the programming side.

Topics:

* TCP / UDP
* Sockets
* Client / Server architecture
* HTTP
* REST APIs
* JSON
* Network communication
* Async networking

### Projects

```text
TCP Chat Application
HTTP Server
REST API
IoT Telemetry Receiver
```

One of the bigger experiments will connect this with my ESP32/IoT work:

```text
ESP32
  ↓
Wi-Fi
  ↓
MQTT / HTTP
  ↓
Python
  ↓
Data Processing
  ↓
Database / Dashboard
```

---

# ⚙️ Concurrency

Understanding how Python handles multiple tasks.

* Threads
* Processes
* Locks
* Queues
* Thread pools
* Multiprocessing
* `asyncio`
* Event loops

Projects:

* Concurrent downloader
* Multithreaded log processor
* Async API client
* Multi-process data processing

The important part here is understanding **when concurrency actually helps** instead of just using threads because they exist.

---

# 🔬 Python Internals

One of the most interesting parts of the repository.

Instead of treating Python as magic, I'll explore what happens underneath.

Topics:

* Python objects
* References
* Mutability
* Memory management
* Reference counting
* Garbage collection
* Stack vs Heap concepts
* Bytecode
* CPython
* Python Virtual Machine

A simple Python program:

```python
x = 10
```

should eventually lead to questions like:

```text
What is x?

Where is the object?

How is it represented?

How does Python execute this?

What does the CPU eventually execute?
```

---

# 🔩 Python + C/C++

Python doesn't exist in isolation.

A major part of the journey is understanding how Python connects with lower-level code.

```text
Python
   ↓
C / C++
   ↓
Native Libraries
   ↓
CPU
```

Topics:

* CPython
* Native extensions
* `ctypes`
* `cffi`
* Shared libraries
* Foreign Function Interfaces

### Experiment

Implement a small computation in C/C++ and call it from Python.

Then compare:

```text
Pure Python
     vs
Native C/C++
```

This should also make it easier to understand what libraries such as **NumPy and PyTorch are doing underneath the surface**.

---

# 📊 Scientific Python

Once the fundamentals are comfortable, I'll move into the Python ecosystem used heavily in AI/ML.

### NumPy

* Arrays
* Dimensions
* Shapes
* Broadcasting
* Vectorization
* Matrix operations
* Linear algebra
* Memory layout

### Pandas

* DataFrames
* Data cleaning
* Filtering
* Grouping
* Missing data
* CSV/JSON datasets

### Matplotlib

* Data visualization
* Experiment visualization
* ML training curves

---

# 🤖 Machine Learning

Python is where the software side of my AI/ML journey starts coming together.

Topics:

* Linear Regression
* Logistic Regression
* KNN
* Decision Trees
* Random Forests
* SVM
* Clustering
* Feature Engineering
* Model Evaluation

Main tools:

```text
NumPy
Pandas
Scikit-learn
Matplotlib
```

---

# 🧮 ML From Scratch

Before relying completely on ML libraries, I'll implement simplified versions of important algorithms myself.

Planned:

* Linear Regression
* Gradient Descent
* Logistic Regression
* K-Means
* KNN
* Basic Neural Network

The goal isn't to recreate Scikit-learn.

It's to understand what is happening behind:

```text
Data
 ↓
Prediction
 ↓
Loss
 ↓
Gradient
 ↓
Parameter Update
 ↓
Repeat
```

---

# 🧠 Deep Learning

Moving from traditional ML into neural networks.

Topics:

* Tensors
* Neural Networks
* Activation Functions
* Forward Propagation
* Backpropagation
* Loss Functions
* Optimizers
* CNNs
* RNNs
* Transformers
* GPU Computing

Primary framework:

**PyTorch**

---

# ⚡ AI / ML Systems

This is where the different parts of the repository start connecting.

Instead of only training a model inside a notebook, I'll explore the complete pipeline:

```text
Data
 ↓
Preprocessing
 ↓
Model
 ↓
Training
 ↓
Evaluation
 ↓
Inference
 ↓
API / Application
 ↓
Deployment
```

Topics:

* Data pipelines
* Model inference
* REST APIs
* Batch inference
* Streaming inference
* Model serving
* Logging
* Monitoring
* Experiment management
* Model optimization

---

# 🚀 Projects

The repository will gradually move from small scripts to complete systems.

### 🟢 Beginner

* Calculator
* Number analyzer
* Unit converter
* Password generator
* CLI quiz
* File organizer

### 🟡 Intermediate

* Expense tracker
* Log analyzer
* CSV processor
* JSON configuration system
* System monitor
* Multithreaded downloader

### 🔵 Systems

* TCP Chat Application
* Python HTTP Server
* Network monitoring tool
* IoT telemetry pipeline
* ESP32 → Python data pipeline

### 🔴 AI / ML

* ML algorithms from scratch
* Dataset preprocessing pipeline
* Image classifier
* Neural network from scratch
* Model inference API
* Real-time AI pipeline
* Edge AI prototype

---

# 🧩 Repository Structure

```text
Python-foundation-to-systems/
│
├── 01-basics/
├── 02-control-flow/
├── 03-data-structures/
├── 04-functions/
├── 05-oop/
├── 06-modules-packages/
│
├── 07-files-data/
├── 08-os/
├── 09-networking/
├── 10-concurrency/
├── 11-python-internals/
├── 12-python-cpp/
│
├── 13-numpy/
├── 14-pandas/
├── 15-machine-learning/
├── 16-ml-from-scratch/
├── 17-deep-learning/
├── 18-ai-systems/
│
├── projects/
├── experiments/
├── benchmarks/
└── notes/
```

The exact structure will evolve as the repository grows.

---

# 🔧 Tools

```text
Python
Git / GitHub
VS Code
Jupyter

NumPy
Pandas
Matplotlib
SciPy
Scikit-learn

PyTorch

FastAPI
Requests
asyncio

pytest
cProfile
timeit
tracemalloc
```

---

# 🧠 How I'm Approaching Python

I'm trying to avoid the usual:

> Learn syntax → import library → build random project → move on.

Instead, the approach is:

```text
Learn
  ↓
Write
  ↓
Break
  ↓
Debug
  ↓
Understand
  ↓
Build
  ↓
Profile
  ↓
Go Deeper
```

Whenever possible, I'll ask:

> **"What is happening underneath this abstraction?"**

For example:

```python
model(x)
```

is not where the learning ends.

Eventually I want to understand:

```text
Python
 ↓
PyTorch
 ↓
Tensor Operations
 ↓
CPU / GPU
 ↓
Memory
 ↓
Kernels
 ↓
Hardware
```

---

# 🎯 End Goal

By the end of this repository, I want to be comfortable using Python for:

* General programming
* DSA & problem solving
* Automation
* Data processing
* OS interaction
* Networking
* Concurrency
* System utilities
* Machine Learning
* Deep Learning
* AI/ML infrastructure
* Performance experiments
* IoT systems
* Edge AI

And most importantly:

**understanding where Python fits inside a larger computing system.**

---

## 🐍 → ⚙️ → 🤖

Python is the foundation.

Systems are the direction.

AI/ML is the application.

The bigger goal is to connect all three.

---
