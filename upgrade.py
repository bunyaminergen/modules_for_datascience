"""
Upgrade / Update all Library and Packages

"""

from __future__ import annotations

from functools import wraps

import decorators

class upgrade_all_libraries:
  
 # importing necessary libraries

 from os import system
 from pandas import read_csv
 from pandas import option_context
 from pandas import DataFrame

 # detect all outdated packages and libraries 

    @time_cal
    def outdated_list() - > DataFrame:



        # changing pip list outdated command to txt file

        system('pip list --outdated > outdated_list.txt')

        # converting txt file to dataframe format and adding some attribitues
   
        data = (read_csv('outdated_list.txt', sep=r'\s+', skiprows=[1])
.style.set_table_attributes("style='display:inline'")
.set_caption('<p style="font-size:125%"><b> Outdated List <b></p>'))
#.set_table_styles([dict(selector = "caption",
#                         props = [("text-align", "center"),
#                                ("font-size", "150%"),
#                                ("color", 'white')])]))
        
        # showing dataframe

        with option_context('display.max_rows', None,):
            display (data)

    # upgrade all outdated packages and libraries 

    @time_cal
    def upgrade_all(removes = None):

        # importing necessary libraries
        
        from os import system
        from pandas import read_csv

        # changing pip list outdated command to txt file

        system('pip list --outdated > outdated_list.txt')

        # converting txt file to dataframe format

        data = read_csv('outdated_list.txt', sep=r'\s+', skiprows=[1])

        # selecting just packages as list format 

        out_list = list(data['Package'])
        
        # control parameter of function

        try:
            if removes != None:
                for i in removes:
                    out_list.remove(i)
            
            # upgrade all packages and libraries function

            for j in out_list:
                print ("\n" + j + "\n")
                !pip  install j  --upgrade
                !pip3 install j  --upgrade

        # control of ValueError in case enter a value outside from list

        except ValueError:
            print ('Please enter a value inside from outdated list')
