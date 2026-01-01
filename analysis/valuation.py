import math




def graham_number(eps, book):
    if not eps or not book or eps <= 0 or book <= 0:
        return None
    return math.sqrt(22.5 * eps * book)




def dcf_value(fcf, growth, discount=0.09, years=10):
    if not fcf or not growth:
        return None
    value = 0
    for t in range(1, years+1):
        value += fcf * (1+growth)**t / (1+discount)**t
    terminal = (fcf * (1+growth)**years) / (discount-0.025)
    return value + terminal / (1+discount)**years