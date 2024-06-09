uint morton3_x(uint x) {
    x = (x | (x << 16u)) & 0x030000FFu;
    x = (x | (x <<  8u)) & 0x0300F00Fu;
    x = (x | (x <<  4u)) & 0x030C30C3u;
    x = (x | (x <<  2u)) & 0x09249249u;
    return x;
}

uint morton3(uvec3 coords) {
    const uint x = morton3_x(coords.x);
    const uint y = morton3_x(coords.y);
    const uint z = morton3_x(coords.z);
    return x | (y << 1) | (z << 2);
}

uint unmorton3_x(uint x) {
    x &= 0x49249249u;
    x = (x | (x >> 2u))  & 0xc30c30c3u;
    x = (x | (x >> 4u))  & 0x0f00f00fu;
    x = (x | (x >> 8u))  & 0xff0000ffu;
    x = (x | (x >> 16u)) & 0x0000ffffu;
    return x;
}

uvec3 unmorton3(uint index) {
    const uint x = unmorton3_x(index);
    const uint y = unmorton3_x(index >> 1u);
    const uint z = unmorton3_x(index >> 2u);
    return uvec3(x, y, z);
}

float min(float x, float y, float z) {
    if (x <= y && x <= z) return x;
    if (y <= z) return y;
    return z;
}

float max(float x, float y, float z) {
    if (x >= y && x >= z) return x;
    if (y >= z) return y;
    return z;
}

uint get_level_offset(uint level) {
    const uint off_0 = 0;
    const uint off_1 = off_0 + 1;
    const uint off_2 = off_1 + (1 << 3);
    const uint off_3 = off_2 + (1 << 6);
    const uint off_4 = off_3 + (1 << 9);
    const uint off_5 = off_4 + (1 << 12);
    const uint off_6 = off_5 + (1 << 15);
    const uint off_7 = off_6 + (1 << 18);
    const uint off_8 = off_7 + (1 << 21);
    const uint off_9 = off_8 + (1 << 24);
    const uint off_10 = off_9 + (1 << 27);
    const uint off_11 = off_10 + (1 << 30);

    switch (level) {
        case 0: return off_0;
        case 1: return off_1;
        case 2: return off_2;
        case 3: return off_3;
        case 4: return off_4;
        case 5: return off_5;
        case 6: return off_6;
        case 7: return off_7;
        case 8: return off_8;
        case 9: return off_9;
        case 10: return off_10;
        case 11: return off_11;
        default: return 0;
    }
}

uint get_grid_width(uint grid_depth) {
    return 1 << (grid_depth - 1);
}
