{{ fullname }}
{{ full_sep }}

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :show-inheritance:

   {% block functions %}
   .. rubric:: Functions

   .. autosummary::
      :toctree:
      :nosignatures:

      {% for item in functions %}
      {{ item }}
      {% endfor %}
   {% endblock %}

   {% block classes %}
   .. rubric:: Classes

   .. autosummary::
      :toctree:
      :nosignatures:

      {% for item in classes %}
      {{ item }
      {% endfor %}
   {% endblock %}

   {% block exceptions %}
   .. rubric:: Exceptions

   .. autosummary::
      :toctree:
      :nosignatures:

      {% for item in exceptions %}
      {{ item }}
      {% endfor %}
   {% endblock %}