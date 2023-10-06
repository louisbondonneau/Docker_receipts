#!/bin/bash

# Test 1 - par_conv_to_tempo1.sh
output=$(bash /cep/lofar/pulsar/ephem_scripts/par_conv_to_tempo1.sh /databf/nenufar-pulsar/ES03/ephem/B1919+21.par)
if [[ "$output" == *"WARNING: TZ mode, ref mjd  not set"* ]] && [[ "$output" == *"B1919+21   25-DEC-18"* ]]; then
    result1="\e[32mPass\e[0m" # Green text for success
else
    result1="\e[31mFail\e[0m" # Red text for failure
fi

# Test 2 - psrchive_info
output=$(psrchive_info)
if [[ "$output" == *"TIMER Archive version 12.3"* ]] && \
   [[ "$output" == *"PSRFITS version 6.7"* ]] && \
   [[ "$output" == *"FFTW3"* ]] && \
   [[ "$output" == *"Tempo2::Predictor support enabled"* ]]; then
    result2="\e[32mPass\e[0m"
else
    result2="\e[31mFail\e[0m"
fi

# Test 3 - import presto
output=$(python -c 'import presto')
if [[ "$output" == "\n" ]]; then
    result3="\e[31mFail\e[0m"
else
    result3="\e[32mPass\e[0m"
fi

# Test 4 - test_presto_python.py
output=$(python /usr/local/pulsar/presto/tests/test_presto_python.py )
if [[ "$output" == *"success"* ]] || [[ "$output" == *"resid2.tmp"* ]]; then
    result4="\e[32mPass\e[0m"
else
    result4="\e[31mFail\e[0m"
fi

# Test 5 - tempo2
cd /usr/local/pulsar/tempo2/example_data
output=$(tempo2 -f example1.par example1.tim  -nofit)
if [[ "$output" == *"Total time span = 2000.588 days = 5.477 years"* ]]; then
    result5="\e[32mPass\e[0m"
else
    result5="\e[31mFail\e[0m"
fi

# Print the summary table with colored text
echo -e "Test                                     Result"
echo -e "TEMPO1 Test Polyco                       $result1"
echo -e "psrchive_info                            $result2"
echo -e "import presto                            $result3"
echo -e "test_presto_python.py                    $result4"
echo -e "tempo2                                   $result5"
