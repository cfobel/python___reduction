#include "reduction.hpp"

using namespace reduction;

{%- if not ops -%}
{%- set ops=["sum", "product", "min", "max"] -%}
{%- endif -%}

{%- if not c_types -%}
{%- set c_types=["float", "int"] -%}
{%- endif -%}

{% for op in ops -%}
{%- for c_type in c_types -%}
extern "C" __global__ void reduce_{{ c_type }}_{{ op }}(int size, {{ c_type }} *data) {
    reduce<{{ c_type }}, {{ op.upper() }} >(size, data);
}
{% endfor -%}
{% endfor %}

{% for op in ops -%}
{%- for c_type in c_types -%}
extern "C" __global__ void global_reduce_{{ c_type }}_{{ op }}(int size, {{ c_type }} *data) {
    global_reduce<{{ c_type }}, {{ op.upper() }}>(size, data, data);
}
{% endfor -%}
{% endfor %}
