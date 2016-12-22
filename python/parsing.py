# -*- coding: utf-8 -*-
#!/bin/python3.5

import re

with open("temp_logging", 'r') as log:
  data = log.read().splitlines()

already_computed = [ re.match(r'.*?\:\s(.*?),\s.*?g\:\s(.*?),\sk\:\s(.*?),\sl_u\:\s(.*?),\sl_i\:\s(.*?)$', line, re.DOTALL).groups() for line in data ]
already_computed = [(float(pred), float(g), int(k), float(l_u), float(l_i)) for pred, g, k, l_u, l_i in already_computed]
already_sorted = sorted(already_computed, key=lambda tup: tup[0])#[0:5]

for best_pred, g, k, lambda_user, lambda_item in already_sorted:
  print("RMSE: {}, with g = {}, k = {}, lambda_user = {} and lambda_item = {}".format(best_pred, g, k, lambda_user, lambda_item))
