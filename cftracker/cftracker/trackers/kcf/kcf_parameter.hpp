// Set the value the same as testing_ECO_gpu.m
#ifndef CFTRACKER_KCF_PARAMETERS_HPP
#define CFTRACKER_KCF_PARAMETERS_HPP

#include "cftracker/common/common.hpp"


namespace cftracker {

struct KcfParameters {
	bool hog =true;
	bool fixed_window = true;
	bool multiscale = true;
	bool lab = true;
};

} // namespace cftracker

#endif // CFTRACKER_KCF_PARAMETERS_HPP


