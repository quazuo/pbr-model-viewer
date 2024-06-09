#pragma once

#include <string>
#include <vector>

struct AutomatonPreset {
    std::uint32_t surviveMask;
    std::uint32_t birthMask;
    std::uint32_t stateCount;
    std::uint32_t useMooreNeighborhood; // not `bool` because of GLSL issues

    [[nodiscard]]
    std::string toString() const;
};

const std::vector<std::pair<std::string, AutomatonPreset> > automatonPresets{
    {
        "Clouds 1", AutomatonPreset{
            .surviveMask = 0x07ffe000u,
            .birthMask = 0x000e6000u,
            .stateCount = 2,
            .useMooreNeighborhood = 1u,
        }
    },
    {
        "Clouds 2", AutomatonPreset{
            .surviveMask = 0x07fff000u,
            .birthMask = 0x00006000u,
            .stateCount = 2,
            .useMooreNeighborhood = 1u,
        }
    },
    {
        "Pulse Waves", AutomatonPreset{
            .surviveMask = 0x00000008u,
            .birthMask = 0x0000000eu,
            .stateCount = 10,
            .useMooreNeighborhood = 1u,
        }
    },
    {
        "Pyroclastic", AutomatonPreset{
            .surviveMask = 0x000000f0u,
            .birthMask = 0x000001c0u,
            .stateCount = 10,
            .useMooreNeighborhood = 1u,
        }
    },
    {
        "Coral", AutomatonPreset{
            .surviveMask = 0x000001e0u,
            .birthMask = 0x000012c0u,
            .stateCount = 4,
            .useMooreNeighborhood = 1u,
        }
    },
    {
        "445", AutomatonPreset{
            .surviveMask = 0x00000010u,
            .birthMask = 0x00000010u,
            .stateCount = 5,
            .useMooreNeighborhood = 1u,
        }
    },
    {
        "Amoeba", AutomatonPreset{
            .surviveMask = 0x07fffe00u,
            .birthMask = 0x0000b0e0u,
            .stateCount = 5,
            .useMooreNeighborhood = 1u,
        }
    },
    {
        "Custom", {}
    }
};

struct AutomatonConfig {
    std::uint32_t gridDepth = 8u;
    AutomatonPreset preset = automatonPresets[0].second;
};
