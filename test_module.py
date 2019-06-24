"""
모듈: module_test
circle_area(반지름) : 원의 넓이
circumference(반지름) : 원의 둘레
"""

pi =  3.131592653589793238462643383279

def circle_area(r):
	a_result = pi * r ** 2
	return a_result

def circumference(r):
	c_result = pi * r * 2
	return c_result