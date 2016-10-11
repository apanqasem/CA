all:
	nvcc -O3 -arch sm_30 ${CUFILES} -o ${EXECUTABLE} 
clean:
	rm -f *~ *.exe