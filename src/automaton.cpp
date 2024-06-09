#include "automaton.h"

#include <sstream>

static std::string maskToString(const std::uint32_t mask, const bool isMoore) {
    std::stringstream ss;
    bool lastWasZero = false;
    bool isFirst = true;
    int lower = -1, higher = -1;

    for (int i = 0; i < (isMoore ? 27 : 7); i++) {
        if (mask & (1 << i)) {
            if (lastWasZero || lower == -1) {
                lower = i;
            }
            higher = i;
            lastWasZero = false;
        } else {
            if (lower != -1 && !lastWasZero) {
                if (!isFirst) {
                    ss << ",";
                } else {
                    isFirst = false;
                }

                ss << lower;
                if (lower < higher) ss << "-" << higher;
            }
            lastWasZero = true;
        }
    }

    if (!lastWasZero) {
        if (!isFirst) ss << ",";
        ss << lower;
        if (lower < higher) ss << "-" << higher;
    }

    return ss.str();
}

std::string AutomatonPreset::toString() const {
    std::stringstream ss;

    ss << maskToString(surviveMask, useMooreNeighborhood)
            << "/" << maskToString(birthMask, useMooreNeighborhood)
            << "/" << stateCount
            << "/" << (useMooreNeighborhood ? "M" : "N");

    return ss.str();
}
