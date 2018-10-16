
all: build test
	
build: admin user
	
test: libgtest.a
	g++ -pthread libgtest.a -I ./googletest/include/ --coverage ./tests/src/*.cpp ./src/controller/*.cpp ./src/interface/*.cpp ./src/model/*.cpp ./src/utils/*.cpp -o test -I ./include/controller/ -I ./include/interface/ -I ./include/model/ -I ./include/utils/ -std=c++11
	./test
	lcov --capture --directory ./ --output-file cov.info
	genhtml cov.info --output-directory html
clean:
	rm ./admin ./user ./test ./*.gcda ./*.gcno
admin:
	g++ ./src/admin_main.cpp ./src/controller/*.cpp ./src/interface/*.cpp ./src/model/*.cpp ./src/utils/*.cpp -o admin -I ./include/controller/ -I ./include/interface/ -I ./include/model/ -I ./include/utils/ -std=c++11

user:
	g++ ./src/user_main.cpp ./src/controller/*.cpp ./src/interface/*.cpp ./src/model/*.cpp ./src/utils/*.cpp -o user -I ./include/controller/ -I ./include/interface/ -I ./include/model/ -I ./include/utils/ -std=c++11

libgtest.a:
	g++ -isystem ./googletest/include -I ./googletest/ -pthread -c ./googletest/src/gtest-all.cc
	ar -rv libgtest.a gtest-all.o

test_parser:
	g++ -pthread libgtest.a -I ./googletest/include/ --coverage ./tests/src/*.cpp ./src/model/*.cpp ./src/utils/CommandParser.cpp -o test -I ./include/controller/ -I ./include/interface/ -I ./include/model/ -I ./include/utils/ -std=c++11
	./test
	lcov --capture --directory ./ --output-file cov.info
	genhtml cov.info --output-directory html
