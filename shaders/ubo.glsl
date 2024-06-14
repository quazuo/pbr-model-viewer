struct WindowRes {
    uint width;
    uint height;
};

struct Matrices {
    mat4 view;
    mat4 proj;
    mat4 inverse_vp;
    mat4 static_view;
};

struct MiscData {
    vec3 camera_pos;
};
