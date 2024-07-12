struct WindowRes {
    uint width;
    uint height;
};

struct Matrices {
    mat4 model;
    mat4 view;
    mat4 proj;
    mat4 inverse_vp;
    mat4 static_view;
    mat4 cubemap_capture_views[6];
    mat4 cubemap_capture_proj;
};

struct MiscData {
    float debug_number;
    vec3 camera_pos;
    vec3 light_direction;
};
