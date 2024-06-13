struct WindowRes {
    uint width;
    uint height;
};

struct Matrices {
    mat4 view;
    mat4 proj;
    mat4 inverse_vp;
};

struct MiscData {
    vec3 camera_pos;
};
