{{ name | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :special-members: __call__, __add__, __mul__, __matmul__

   {% block methods %}
   {% set own = methods | reject("in", inherited_from_torch) | reject("eq", "__init__") | list %}
   {% if own %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
      :nosignatures:
   {% for item in own %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
