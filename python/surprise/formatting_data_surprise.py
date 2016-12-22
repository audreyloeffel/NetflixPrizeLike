# -*- coding: utf-8 -*-
#!/bin/python3.5

import re

def format_data():
  """
  Create a data file with compliance to the surprise library.
  """
  indices = []
  with open("../../data/data_train.csv", 'r') as data_file:
    data = data_file.read().splitlines()[1:]
  indices = [ re.match(r'r(\d+?)_c(\d+?),(\d)$', line, re.DOTALL).groups() for line in data ]

  hundred_percent = len(indices)
  counter = 0
  with open('../../data/data_set.data', 'w') as data_file:
    for item, user, rating in indices:
      data_file.write('\t'.join([user, item, rating]) + '\n')
      counter += 1
      print("\rProgress: {}%".format(int(counter/hundred_percent*100)), end='')

if __name__ == '__main__':
  print("Formatting the file in .data for surprise to use")
  format_data()
  print("\nDone")
