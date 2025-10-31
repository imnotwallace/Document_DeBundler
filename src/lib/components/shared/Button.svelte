<script lang="ts">
  import type { ButtonProps } from '../../types';

  // Props with defaults
  export let variant: ButtonProps['variant'] = 'primary';
  export let size: ButtonProps['size'] = 'md';
  export let disabled: boolean = false;
  export let type: ButtonProps['type'] = 'button';

  // Variant styles
  const variantClasses = {
    primary: 'bg-blue-600 hover:bg-blue-700 text-white focus:ring-blue-500',
    secondary: 'bg-gray-600 hover:bg-gray-700 text-white focus:ring-gray-500',
    success: 'bg-green-600 hover:bg-green-700 text-white focus:ring-green-500',
    danger: 'bg-red-600 hover:bg-red-700 text-white focus:ring-red-500',
    disabled: 'bg-gray-400 cursor-not-allowed text-gray-200',
  };

  // Size styles
  const sizeClasses = {
    sm: 'py-1.5 px-3 text-sm',
    md: 'py-2.5 px-4 text-base',
    lg: 'py-3.5 px-6 text-lg',
  };

  // Base classes
  const baseClasses =
    'font-medium rounded-lg transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed';

  // Compute final classes
  $: classes = `${baseClasses} ${sizeClasses[size || 'md']} ${
    disabled ? variantClasses.disabled : variantClasses[variant || 'primary']
  }`;
</script>

<button
  {type}
  {disabled}
  class={classes}
  on:click
>
  <slot />
</button>

<style>
  /* Additional custom styles can go here if needed */
</style>
