#ifndef TESTS_H
#define TESTS_H

#include <string>

bool run_cpu_reference_test(const std::string &kernel,
                            const std::string &image_path,
                            const std::string &filter_path);

#endif
