using System;
using System.Threading.Tasks;
using System.Windows.Input;

namespace Ecg.Classification.Client.Commands
{
	public interface IAsyncCommand : ICommand
	{
		Task ExecuteAsync(object parameter);
	}

	public class AsyncCommand<T> : IAsyncCommand
	{
		private readonly Func<T, Task> _execute = null;
		private readonly Predicate<T> _canExecute = null;

		public AsyncCommand(Func<T, Task> execute) : this(execute, null)
		{
		}

		public AsyncCommand(Func<T, Task> execute, Predicate<T> canExecute)
		{
			_execute = execute ?? throw new ArgumentNullException(nameof(execute));
			_canExecute = canExecute;
		}

		public bool CanExecute(object parameter)
		{
			return _canExecute == null ? true : _canExecute((T)parameter);
		}

		public event EventHandler CanExecuteChanged
		{
			add { CommandManager.RequerySuggested += value; }
			remove { CommandManager.RequerySuggested -= value; }
		}

		public void Execute(object parameter)
		{
			ExecuteAsync(parameter);
		}

		public async Task ExecuteAsync(object parameter)
		{
			if (parameter is T parsedParameter)
			{
				await _execute((T)parameter);
			}
		}
	}

	public class AsyncCommand : IAsyncCommand
	{
		private readonly Func<Task> _execute;
		private readonly Func<bool> _canExecute;

		public AsyncCommand(Func<Task> execute) : this(execute, null)
		{
		}

		public AsyncCommand(Func<Task> execute, Func<bool> canExecute)
		{
			_execute = execute ?? throw new ArgumentNullException(nameof(execute));
			_canExecute = canExecute;
		}

		public bool CanExecute(object parameter)
		{
			return _canExecute == null ? true : _canExecute();
		}

		public event EventHandler CanExecuteChanged
		{
			add { CommandManager.RequerySuggested += value; }
			remove { CommandManager.RequerySuggested -= value; }
		}

		public void Execute(object parameter)
		{
			ExecuteAsync(parameter);
		}

		public async Task ExecuteAsync(object parameter)
		{
			await _execute();
		}
	}
}
